from utils import *
import torch.utils.data as data
from model import Model

if __name__ == '__main__':
    model = torch.load(MODEL_DIR+'model_30.pth', map_location=DEVICE, weights_only=False)

    dataset = Dataset('test')
    loader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=dataset.collate_fn)

    with torch.no_grad():

        correct_cnt = pred_cnt = gold_cnt = 0

        for b, batch in enumerate(loader):
            input_ids, mask, ent_label, ent_cdm, ent_cdw, pola_label, pairs = batch

            input_ids = input_ids.to(DEVICE)
            mask = mask.to(DEVICE)
            ent_label = ent_label.to(DEVICE)
            ent_cdm = ent_cdm.to(DEVICE)
            ent_cdw = ent_cdw.to(DEVICE)
            pola_label = pola_label.to(DEVICE)

            # Entity part
            pred_ent_label = model.get_entity(input_ids, mask)

            # Polarity part
            pred_pola = model.get_pola(input_ids, mask, ent_cdm, ent_cdw)

            # Loss calculation
            loss = model.loss_fn(input_ids, ent_label, mask, pred_pola, pola_label)

            # if b % 10 == 0:
            print('>> batch:', b, 'loss:', loss.item())

            # if b % 100 != 0:
            #     continue

            # Calculate accuracy (both entity and sentiment must be correct to count)
            for i in range(len(input_ids)):
                # Accumulate ground truth count
                gold_cnt += len(pairs[i])

                # Parse entity positions from predicted entity labels and predict sentiment
                b_ent_pos, b_ent_pola = get_pola(model, input_ids[i], mask[i], pred_ent_label[i])
                if not b_ent_pos:
                    continue

                # Parse entities and sentiments, compare with ground truth
                pred_pair = []
                cnt = 0
                for ent, pola in zip(b_ent_pos, torch.argmax(b_ent_pola, dim=1)):
                    pair_item = (ent, pola.item())
                    pred_pair.append(pair_item)
                    # If correct, increment correct count
                    if pair_item in pairs[i]:
                        cnt += 1

                # Accumulate counts
                correct_cnt += cnt
                pred_cnt += len(pred_pair)

        # Metrics calculation
        precision = round(correct_cnt / (pred_cnt + EPS), 3)
        recall = round(correct_cnt / (gold_cnt + EPS), 3)
        f1_score = round(2 / (1 / (precision + EPS) + 1 / (recall + EPS)), 3)
        print('\tcorrect_cnt:', correct_cnt, 'pred_cnt:', pred_cnt, 'gold_cnt:', gold_cnt)
        print('\tprecision:', precision, 'recall:', recall, 'f1_score:', f1_score)

