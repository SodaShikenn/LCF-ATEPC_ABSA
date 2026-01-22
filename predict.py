from utils import *
from model import *
from transformers import BertTokenizer

if __name__ == '__main__':
    model = torch.load(MODEL_DIR+'model_30.pth', map_location=DEVICE, weights_only=False)

    with torch.no_grad():

        text = '这个手机外观时尚，美中不足的是拍照像素低。'

        tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
        tokens = ['[CLS]'] + list(text) + ['[SEP]']
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        mask = [1] * len(input_ids)
        
        # Entity part
        input_ids = torch.tensor(input_ids).unsqueeze(0)
        mask = torch.tensor(mask).unsqueeze(0).bool()

        input_ids = input_ids.to(DEVICE)
        mask = mask.to(DEVICE)

        pred_ent_label = model.get_entity(input_ids, mask)
        
        # Polarity part
        b_ent_pos, b_ent_pola = get_pola(model, input_ids[0], mask[0], pred_ent_label[0])

        if not b_ent_pos:
            print('\t', 'no result.')
        else:
            pred_pair = []
            for ent_pos, pola in zip(b_ent_pos, torch.argmax(b_ent_pola, dim=1)):
                aspect = text[ent_pos[0] - 1:ent_pos[-1]]
                pred_pair.append({'aspect': aspect, 'sentiment': POLA_MAP[pola.item()], 'position': ent_pos})

            print('\t', text)
            print('\t', pred_pair)