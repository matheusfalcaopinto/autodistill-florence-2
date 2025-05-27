import os
from dataclasses import dataclass

import cv2
import numpy as np
import supervision as sv
import torch
from autodistill.detection import (CaptionOntology, DetectionBaseModel,
                                  DetectionTargetModel)
from autodistill.helpers import load_image
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
# CORREÇÃO 1: Importação do AdamW corrigida
from torch.optim import AdamW
from transformers import (AutoModelForCausalLM, AutoProcessor,
                          get_scheduler)

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# A classe DetectionsDataset permanece a mesma, pois sua lógica está correta.
class DetectionsDataset(Dataset):
    def __init__(self, dataset: sv.DetectionDataset):
        self.dataset = dataset
        self.keys = list(dataset.images.keys())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        key = self.keys[idx]
        image = self.dataset.images[key]
        annotations = self.dataset.annotations[key]
        h, w, _ = image.shape

        boxes = (annotations.xyxy / np.array([w, h, w, h]) * 1000).astype(int).tolist()
        labels = [self.dataset.classes[class_id] for class_id in annotations.class_id]

        prefix = "<OD>"

        suffix_components = []
        for [x1, y1, x2, y2], label in zip(boxes, labels):
            suffix_component = f"{label}<loc_{x1}><loc_{y1}><loc_{x2}><loc_{y2}>"
            suffix_components.append(suffix_component)

        suffix = "".join(suffix_components)
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return prefix, suffix, image


# -----------------------------------------------------------------------------
# CLASSE DE INFERÊNCIA (Pós-treino) REFATORADA
# -----------------------------------------------------------------------------

@dataclass
class Florence2(DetectionBaseModel):
    ontology: CaptionOntology

    # MELHORIA: O construtor agora só precisa do caminho para o modelo treinado (adaptadores LoRA)
    def __init__(self, ontology: CaptionOntology, model_path: str):
        """
        Inicializa o modelo de inferência a partir de um checkpoint PEFT (LoRA) treinado.

        Args:
            ontology (CaptionOntology): A ontologia para o autodistill.
            model_path (str): Caminho para a pasta contendo os adaptadores LoRA salvos
                                e o processador (ex: "./final_model_peft/").
        """
        self.ontology = ontology

        # Carrega o processador do mesmo local que os adaptadores
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        # Para carregar um modelo com adaptadores LoRA, primeiro carregamos o modelo base
        # e depois aplicamos os adaptadores. O ID do modelo base deve ser o mesmo usado no treino.
        base_model_id = "microsoft/Florence-2-large-ft" # Ou 'base-ft', dependendo do treino
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            trust_remote_code=True,
            device_map="auto", # 'auto' é mais flexível que 'cuda'
            torch_dtype=torch.bfloat16 # Usar bfloat16 para eficiência
        ).eval()

        # Agora, carrega os adaptadores LoRA sobre o modelo base
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(base_model, model_path).eval()
        print("Modelo PEFT carregado para inferência.")


    def predict(self, input: str, confidence: int = 0.5) -> sv.Detections:
        image = load_image(input, return_format="PIL")
        
        # CORREÇÃO: Usar a task <OD> que é consistente com o fine-tuning.
        task_prompt = "<OD>"

        # Inferência
        inputs = self.processor(text=task_prompt, images=image, return_tensors="pt").to(DEVICE)
        
        # Usamos bfloat16 para inferência mais rápida
        with torch.autocast("cuda", dtype=torch.bfloat16):
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"].to(torch.bfloat16),
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False
            )
        
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        # Pós-processamento para extrair caixas e rótulos
        parsed_answer = self.processor.post_process_generation(
            generated_text, task=task_prompt, image_size=(image.width, image.height)
        )

        # CORREÇÃO: Usar o helper do Supervision para extrair caixas E PONTUAÇÕES
        # Isso evita o bug de ter confiança 1.0 para tudo.
        detections = sv.Detections.from_lmm(
            lmm=sv.LMM.FLORENCE_2,
            result=parsed_answer,
            resolution_wh=image.size
        )
        
        # Filtrar pela ontologia e pelo limiar de confiança
        ontology_classes = self.ontology.classes()
        
        # Mapeia os labels detectados para os IDs da ontologia
        final_detections_mask = np.array([label in ontology_classes for label in detections.data['class_name']], dtype=bool)
        detections = detections[final_detections_mask]

        if len(detections) > 0:
            class_ids = np.array([ontology_classes.index(label) for label in detections.data['class_name']])
            detections.class_id = class_ids

        return detections[detections.confidence > confidence]


# -----------------------------------------------------------------------------
# CLASSE DE TREINAMENTO REFATORADA
# -----------------------------------------------------------------------------

class Florence2Trainer(DetectionTargetModel):
    # MELHORIA: Aceita o ID do modelo base como parâmetro
    def __init__(
        self,
        model_id: str = "microsoft/Florence-2-large-ft",
    ):
        # CORREÇÃO: Removida a `revision` inválida.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Carrega o modelo e o processador usando o ID fornecido
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True
        )
        self.model_id = model_id # Salva para referência
        print(f"Modelo base '{model_id}' carregado no dispositivo: {self.device}")

    def predict(self, input: str, confidence=0.5) -> sv.Detections:
        # A lógica de predição aqui seria com o modelo *pós-treino*, que é o objetivo da classe Florence2.
        # Esta função é mais um placeholder para conformidade com a classe base.
        # A inferência real deve ser feita com a classe `Florence2` após salvar o modelo.
        raise NotImplementedError("Use a classe `Florence2` para inferência após o treino.")

    def train(self, dataset_path, epochs=10, lr=5e-6, batch_size=4):
        ds_train = sv.DetectionDataset.from_coco(
            images_directory_path=f"{dataset_path}/train",
            annotations_path=f"{dataset_path}/train/_annotations.coco.json",
        )

        ds_valid = sv.DetectionDataset.from_coco(
            images_directory_path=f"{dataset_path}/valid",
            annotations_path=f"{dataset_path}/valid/_annotations.coco.json",
        )

        train_dataset = DetectionsDataset(ds_train)
        val_dataset = DetectionsDataset(ds_valid)

        def collate_fn(batch):
            questions, answers, images = zip(*batch)
            inputs = self.processor(
                text=list(questions),
                images=list(images),
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            return inputs, answers

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, collate_fn=collate_fn
        )

        # Configuração do LoRA
        config = LoraConfig(
            r=8,
            lora_alpha=8,
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "linear", "Conv2d", "lm_head", "fc2"],
            task_type="CAUSAL_LM",
            lora_dropout=0.05,
            bias="none",
            inference_mode=False,
            use_rslora=True,
            init_lora_weights="gaussian",
        )
        # Cria o modelo PEFT para treinamento
        peft_model = get_peft_model(self.model, config)
        peft_model.print_trainable_parameters()
        
        torch.cuda.empty_cache()

        # CORREÇÃO: O otimizador deve atuar sobre os parâmetros do `peft_model`
        optimizer = AdamW(peft_model.parameters(), lr=lr)
        num_training_steps = epochs * len(train_loader)
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )

        for epoch in range(epochs):
            # Mude o modelo para o modo de treino
            peft_model.train()
            train_loss = 0
            for inputs, answers in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
                labels = self.processor.tokenizer(
                    text=answers,
                    return_tensors="pt",
                    padding=True,
                    return_token_type_ids=False,
                ).input_ids.to(self.device)
                
                # CORREÇÃO: Use `peft_model` para o forward pass
                outputs = peft_model(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    labels=labels
                )
                loss = outputs.loss

                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            print(f"Average Training Loss: {avg_train_loss}")

            # Validação
            peft_model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, answers in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}"):
                    labels = self.processor.tokenizer(
                        text=answers, return_tensors="pt", padding=True
                    ).input_ids.to(self.device)
                    
                    # CORREÇÃO: Use `peft_model` para o forward pass
                    outputs = peft_model(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        labels=labels
                    )
                    loss = outputs.loss
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            print(f"Average Validation Loss: {avg_val_loss}")

        # Salva o modelo treinado (apenas os adaptadores LoRA)
        output_dir = "./final_model_peft"
        os.makedirs(output_dir, exist_ok=True)
        # CORREÇÃO: Salve o `peft_model`, não o `self.model`
        peft_model.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)
        print(f"Adaptadores LoRA e processador salvos em: {output_dir}")
