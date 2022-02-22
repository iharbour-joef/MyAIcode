# 字节跳动安全AI竞赛个人代码整理
竞赛网址：https://security.bytedance.com/fe/ai-challenge#/challenge#schedule

赛道方向：基于文本和多模态数据的风险识别

# 题目：电商黄牛地址识别
赛题描述：

    电商活动过程中，平台一般会为某些特定商品提供补贴/优惠券。补贴后的商品价格将会低于
    市场价格，存在套利空间，也会变成黄牛的重灾区。常见的补贴商品包括iPhone，黄金，茅台
    白酒等。黄牛团伙会召集多个用户抢购某一些补贴商品，收货后再以高于购买价的价格售卖给其他渠道。

评估标准：

    风控模型会更加关注策略的准确率，如果准确率不足则会导致大规模误伤；评估基本思想是
    保证准确性极高以上尽可能提高召回率。因此我们本次采用F-beta，beta值选择为0.3，
    F-beta：(1+ beta ** 2) * Precision * Recall / (beta **2 * Precision + Recall)  
    按照 F-beta 倒序排名，排名较高者优胜。
    
# 题目：色情导流用户识别
赛题描述：

    本次比赛的主要目的是以端到端的方式对色情导流用户进行识别：
    输入：用户的特征，包括基础信息、投稿信息、行为信息。
    输出：用户的标签（1表示色情导流用户，0表示正常用户）

评估标准：

    在实际风控业务中，色情导流用户的数量与正常用户相比有较大差距，如果准确率较低会
    召回大量正常用户，因此本赛题采用F-beta对模型进行评估。beta等于0.3。
