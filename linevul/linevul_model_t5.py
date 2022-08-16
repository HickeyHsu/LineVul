import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import T5ForConditionalGeneration

class T5ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks.
    2个线性层组成的全连接层
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.out_proj = nn.Linear(config.d_model, 2)

    def forward(self, features, **kwargs):# dropout->线性层->dropout->线性层
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)#激活函数tanh
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
        
class Model(T5ForConditionalGeneration):
    """ 
    lineVul模型:基于Roberta
    """   
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__(config=config)
        self.encoder:T5ForConditionalGeneration = encoder#编码器
        self.tokenizer = tokenizer#
        self.classifier = T5ClassificationHead(config)#下游任务：2分类
        self.args = args
    
    # 前向传播    
    def forward(self, input_embed=None, labels=None, output_attentions=False, input_ids=None):
        if output_attentions:#是否输出注意力分数（训练时不需要)
            if input_ids is not None:#
                """
                如果输入input_ids,即没有进行词嵌入，在这里进行mask和词嵌入
                要将句子处理为特定的长度，就要在句子前或后补[PAD]
                """
                outputs = self.encoder.encoder(input_ids, attention_mask=input_ids.ne(0), output_attentions=output_attentions)#相当于调用encoder的__call__方法，进行前向传播
            else:#已经进行词嵌入的情况
                outputs = self.encoder.encoder(inputs_embeds=input_embed, output_attentions=output_attentions)
            attentions = outputs.attentions#注意力
            # 上游的最后一层结果输入到下游分类器
            last_hidden_state = outputs.last_hidden_state
            logits = self.classifier(last_hidden_state)
            # 分类器结果用softmax进行分类
            prob = torch.softmax(logits, dim=-1)
            #如果有标签（验证集）计算交叉熵损失
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels)
                return loss, prob, attentions
            else:
                return prob, attentions
        else:
            if input_ids is not None:
                outputs = self.encoder.roberta(input_ids, attention_mask=input_ids.ne(1), output_attentions=output_attentions)[0]
            else:
                outputs = self.encoder.roberta(inputs_embeds=input_embed, output_attentions=output_attentions)[0]
            logits = self.classifier(outputs)
            prob = torch.softmax(logits, dim=-1)
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels)
                return loss, prob
            else:
                return prob