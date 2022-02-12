from transformers import T5Tokenizer
from inference.T5FineTunedModel import T5FineTuner

MODEL_NAME = 't5-base'
TOKENIZER = T5Tokenizer.from_pretrained(MODEL_NAME)
T5_model = T5FineTuner()
TRAINED_MODEL = T5_model.load_from_checkpoint('..\\training\\lightning_logs\\fine_tuning_text_summarizer_rt_v_0_2\\version_0\\checkpoints\\epoch=19-step=4319.ckpt')
MAX_LEN = 75

def custom_summarize(text = '') -> str: 
    '''
    Summarizes text using trained model, tokenizer, and desired summary text
    '''
        
    text_encoding = TOKENIZER(
        text,
        max_length = MAX_LEN, 
        padding = 'max_length', 
        truncation = True, 
        return_attention_mask = True, 
        return_tensors = 'pt'
    )
    generated_ids = TRAINED_MODEL.model.generate(
        input_ids=text_encoding['input_ids'], 
        attention_mask = text_encoding['attention_mask'], 
        max_length = MAX_LEN,
        num_beams = 2,
        repetition_penalty = 2.5,
        length_penalty = 1.0,
    )
    preds = [
        TOKENIZER.decode(gen_id, 
            skip_special_tokens = True, 
            clean_up_tokenization_spaces = True)
        for gen_id in generated_ids
    ]
    return "".join(preds)