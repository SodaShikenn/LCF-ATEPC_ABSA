from utils import *
import torch.utils.data as data
from model import Model
from tqdm import tqdm

if __name__ == '__main__':
    model = Model().to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    dataset = Dataset()
    loader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=dataset.collate_fn)

    for e in range(EPOCH):
        epoch_loss = 0
        batch_pbar = tqdm(enumerate(loader), total=len(loader), desc=f'Epoch {e}/{EPOCH}')

        for b, batch in batch_pbar:
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

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Calculate accuracy (both entity and sentiment must be correct)
            correct_cnt = pred_cnt = gold_cnt = 0
            for i in range(len(input_ids)):
                # Accumulate gold count
                gold_cnt += len(pairs[i])

                # Parse entity positions from predicted label and predict sentiment
                b_ent_pos, b_ent_pola = get_pola(model, input_ids[i], mask[i], pred_ent_label[i])
                if not b_ent_pos:
                    continue

                # Parse entity and sentiment, compare with gold
                pred_pair = []
                cnt = 0
                for ent, pola in zip(b_ent_pos, torch.argmax(b_ent_pola, dim=1)):
                    pair_item = (ent, pola.item())
                    pred_pair.append(pair_item)
                    # Correct prediction, increment count
                    if pair_item in pairs[i]:
                        cnt += 1

                # Accumulate counts
                correct_cnt += cnt
                pred_cnt += len(pred_pair)

            # Metrics calculation
            precision = round(correct_cnt / (pred_cnt + EPS), 3)
            recall = round(correct_cnt / (gold_cnt + EPS), 3)
            f1 = round(2 / (1 / (precision + EPS) + 1 / (recall + EPS)), 3)
            batch_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'P': precision, 'R': recall, 'F1': f1})

        # Epoch summary
        avg_loss = epoch_loss / len(loader)
        tqdm.write(f'Epoch {e}: avg_loss={avg_loss:.4f} | P={precision} R={recall} F1={f1}')

        if e % 10 == 0:
            torch.save(model, MODEL_DIR + f'model_{e}.pth')
