import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import matplotlib.pyplot as plt

def train(train_iter, dev_iter, model, args):
    if args.cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    steps = 0
    best_acc = 0
    last_step = 0
    train_losses = []
    dev_losses = []
    dev_accuracies = []

    for epoch in range(1, args.epochs + 1):
        for batch in train_iter:
            model.train()
            feature, target = batch.text, batch.label
            feature.t_(), target.sub_(1)  # batch first, index align
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()
            logit = model(feature)
            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()

            steps += 1
            train_losses.append(loss.item())  # 记录训练损失
            if steps % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects / batch.batch_size
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                                                                             loss.item(),
                                                                             accuracy.item(),
                                                                             corrects.item(),
                                                                             batch.batch_size))
            if steps % args.test_interval == 0:
                dev_loss, dev_acc = eval(dev_iter, model, args)
                dev_losses.append(dev_loss)
                dev_accuracies.append(dev_acc)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if args.save_best:
                        save(model, args.save_dir, 'best', steps)
                else:
                    if steps - last_step >= args.early_stop:
                        print('early stop by {} steps.'.format(args.early_stop))
                        break
            elif steps % args.save_interval == 0:
                save(model, args.save_dir, 'snapshot', steps)

    print('\nBest accuracy: {:.4f}%'.format(best_acc))
    plot_results(train_losses, dev_losses, dev_accuracies)  # 新增

def eval(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    with torch.no_grad():
        for batch in data_iter:
            feature, target = batch.text, batch.label
            feature.t_(), target.sub_(1)  # batch first, index align
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            logit = model(feature)
            loss = F.cross_entropy(logit, target, reduction='sum')

            avg_loss += loss.item()
            corrects += (torch.max(logit, 1)
                         [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects / size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))
    return avg_loss, accuracy

def predict(text, model, text_field, label_field, cuda_flag):
    assert isinstance(text, str)
    model.eval()
    # text = text_field.tokenize(text)
    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = torch.tensor(text)
    x = autograd.Variable(x)
    if cuda_flag:
        x = x.cuda()
    print(x)
    output = model(x)
    _, predicted = torch.max(output, 1)
    return label_field.vocab.itos[predicted.item() + 1]

def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)

def plot_results(train_losses, dev_losses, dev_accuracies):
    epochs = range(1, len(dev_losses) + 1)
    plt.figure(figsize=(10, 5))


    # 绘制准确率图
    plt.plot(epochs, [x.cpu() for x in dev_accuracies], 'b', label='Validation accuracy')
    plt.ylim(0, 100)  # 设置y轴的范围为0到100
    plt.title('Validation accuracy over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
