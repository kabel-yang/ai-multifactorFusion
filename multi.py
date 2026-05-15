"""
A 股 AI 量化交易系统 (生产级架构)
核心特性：
1. 使用 FinBERT 进行专业金融情感分析
2. LightGBM + 贝叶斯优化模型
3. 基本面 + 技术面 + 舆情多因子融合
4. 完整回测框架 (支持交易成本、滑点)
5. 行业/风格中性化
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 深度学习/NLP
import torch

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 金融数据处理
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.model_selection import TimeSeriesSplit
import talib

# 机器学习
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from bayes_opt import BayesianOptimization
from sklearn.metrics import roc_auc_score

# 深度学习/NLP
#import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import jieba
jieba.initialize()

# 回测引擎
import backtrader as bt

# 配置
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ============================ 1. 数据获取模块 ============================
class DataFetcher:
    """统一数据获取接口 (支持模拟/真实数据)"""
    
    def __init__(self, data_source='simulated'):
        self.data_source = data_source
        
    def fetch_price_data(self, start_date, end_date, universe=None):
        """获取价格数据"""
        if self.data_source == 'simulated':
            return self._generate_simulated_prices(start_date, end_date, universe)
        else:
            # 这里可以接入真实数据源 (akshare, tushare, baostock等)
            raise NotImplementedError("真实数据源需自行实现")
    
    def fetch_fundamental_data(self, dates, universe):
        """获取基本面数据"""
        return self._generate_simulated_fundamentals(dates, universe)
    
    def fetch_news_data(self, start_date, end_date, universe):
        """获取新闻数据"""
        return self._generate_simulated_news(start_date, end_date, universe)
    
    def fetch_industry_data(self, universe):
        """获取行业分类数据"""
        return self._generate_simulated_industry(universe)
    
    def _generate_simulated_prices(self, start_date, end_date, universe):
        """生成模拟价格数据 (带市场特性)"""
        if universe is None:
            universe = [f"{i:06d}.SZ" if i < 300000 else f"{i:06d}.SH" 
                      for i in range(1, 51)]
        
        dates = pd.date_range(start_date, end_date, freq='B')
        
        all_data = []
        for code in universe:
            # 生成基础价格序列
            n_days = len(dates)
            
            # 1. 生成随机游走
            np.random.seed(hash(code) % 10000)
            base_returns = np.random.normal(0.0002, 0.02, n_days)
            
            # 2. 添加市场相关性 (beta)
            market_returns = np.random.normal(0.0001, 0.015, n_days)
            beta = np.random.uniform(0.8, 1.2)
            base_returns = 0.7 * base_returns + 0.3 * beta * market_returns
            
            # 3. 添加个股特性
            # 动量效应
            momentum = np.random.uniform(-0.1, 0.1)
            base_returns = base_returns + momentum * 0.1
            
            # 波动率聚集
            for i in range(1, n_days):
                if abs(base_returns[i-1]) > 0.03:  # 大幅波动后继续波动
                    base_returns[i] *= 1.5
            
            # 计算价格
            base_price = np.random.uniform(10, 100)
            cum_returns = np.cumsum(base_returns)
            prices = base_price * np.exp(cum_returns)
            
            # 添加跳空缺口
            for i in range(0, n_days, 20):
                if i < n_days - 1:
                    gap = np.random.choice([-0.05, -0.02, 0.02, 0.05])
                    prices[i+1:] *= (1 + gap)
            
            for i, date in enumerate(dates):
                all_data.append({
                    'date': date,
                    'code': code,
                    'open': prices[i] * (1 + np.random.uniform(-0.01, 0.01)),
                    'high': max(prices[i] * (1 + np.random.uniform(0, 0.03)), 
                               prices[i]),
                    'low': min(prices[i] * (1 + np.random.uniform(-0.03, 0)), 
                              prices[i]),
                    'close': prices[i],
                    'volume': int(np.random.lognormal(13, 1.5)),
                    'turnover': np.random.uniform(0.5, 8)
                })
        
        df = pd.DataFrame(all_data)
        df['returns'] = df.groupby('code')['close'].pct_change()
        return df
    
    def _generate_simulated_fundamentals(self, dates, universe):
        """生成模拟基本面数据"""
        fundamental_data = []
        
        for code in universe:
            for date in dates:
                # 生成财务指标
                pe = np.random.lognormal(2.5, 0.8)  # PE
                pb = np.random.lognormal(1.0, 0.5)  # PB
                roe = np.random.normal(0.08, 0.05)  # ROE
                debt_ratio = np.random.beta(2, 5)  # 负债率
                
                fundamental_data.append({
                    'date': date,
                    'code': code,
                    'pe': pe,
                    'pb': pb,
                    'roe': roe,
                    'debt_ratio': debt_ratio,
                    'gross_margin': np.random.beta(3, 3),  # 毛利率
                    'revenue_growth': np.random.normal(0.1, 0.15),  # 营收增长
                    'profit_growth': np.random.normal(0.08, 0.2)  # 利润增长
                })
        
        return pd.DataFrame(fundamental_data)
    
    def _generate_simulated_news(self, start_date, end_date, universe):
        """生成模拟新闻数据 (带真实语义)"""
        import random
        
        news_templates = [
            ("{company}发布{year}年年度报告，净利润同比增长{growth}%", "positive"),
            ("{company}获得{industry}领域{amount}亿元订单", "positive"),
            ("{company}与{partner}达成战略合作", "positive"),
            ("{company}新产品{product}正式发布", "positive"),
            ("{company}遭证监会立案调查", "negative"),
            ("{company}{year}年业绩预告，净利润下滑{growth}%", "negative"),
            ("{company}股东拟减持不超过{percent}%股份", "negative"),
            ("{company}发布风险提示公告", "negative"),
            ("{company}召开临时股东大会", "neutral"),
            ("券商给予{company}{rating}评级", "positive"),
        ]
        
        company_names = ["科技", "医药", "新能源", "制造", "消费", "金融", "地产", "传媒"]
        industries = ["人工智能", "生物医药", "新能源汽车", "高端制造", "白酒", "银行", "房地产", "游戏"]
        partners = ["腾讯", "阿里巴巴", "华为", "百度", "字节跳动", "美团"]
        products = ["智能芯片", "创新药", "锂电池", "工业机器人", "高端白酒", "金融科技产品"]
        
        news_data = []
        dates = pd.date_range(start_date, end_date, freq='D')
        
        for _ in range(1000):  # 生成1000条新闻
            code = random.choice(universe)
            date = random.choice(dates)
            template, sentiment = random.choice(news_templates)
            
            # 填充模板
            content = template.format(
                company=f"{code[:6]}公司",
                year=date.year,
                growth=random.randint(5, 50) if "增长" in template else random.randint(-30, -5),
                amount=random.randint(1, 20),
                industry=random.choice(industries),
                partner=random.choice(partners),
                product=random.choice(products),
                percent=random.randint(1, 3),
                rating=random.choice(["买入", "增持", "强烈推荐"])
            )
            
            # 添加详细内容
            details = [
                "公司表示，此次合作将进一步提升市场竞争力。",
                "分析师认为，该事件对公司长期发展有积极影响。",
                "市场人士指出，需关注相关风险因素。",
                "公司管理层对未来发展充满信心。"
            ]
            
            content += random.choice(details)
            
            news_data.append({
                'timestamp': pd.Timestamp(f"{date.date()} {random.randint(9, 15)}:{random.randint(0, 59)}:00"),
                'date': date.date(),
                'code': code,
                'title': f"关于{code[:6]}的重要公告",
                'content': content,
                'source': random.choice(['交易所公告', '公司公告', '媒体报道', '券商研报']),
                'url': f"http://example.com/news/{random.randint(10000, 99999)}",
                'sentiment': sentiment
            })
        
        return pd.DataFrame(news_data).sort_values('timestamp')
    
    def _generate_simulated_industry(self, universe):
        """生成模拟行业数据"""
        industries = ['银行', '非银金融', '医药生物', '电子', '计算机', 
                     '通信', '食品饮料', '家用电器', '汽车', '电力设备']
        
        industry_data = []
        for code in universe:
            industry = np.random.choice(industries, p=[0.1, 0.1, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05])
            industry_data.append({
                'code': code,
                'industry': industry
            })
        
        return pd.DataFrame(industry_data)

# ============================ 2. NLP 情感分析模块 ============================    
class AdvancedSentimentAnalyzer:
    """高级情感分析器 (FinBERT + 规则增强)"""
    def __init__(self, model_name='yiyanghkust/finbert-tone'):
        """
        使用预训练的 FinBERT 模型
        修复：明确使用BertTokenizer而不是AutoTokenizer
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 尝试加载 FinBERT
        self.use_bert = False
        try:
            # 修复：明确使用BertTokenizer，设置use_fast=False强制使用慢速tokenizer
            from transformers import BertTokenizer, BertForSequenceClassification
            
            print("正在加载BERT模型...")
            self.tokenizer = BertTokenizer.from_pretrained(
                model_name, 
                use_fast=False  # 强制使用慢速tokenizer
            )
            self.model = BertForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            self.use_bert = True
            print("成功加载BERT模型（使用慢速tokenizer）")
        except Exception as e:
            print(f"无法加载BERT模型，将使用词典方法: {e}")
            self.use_bert = False
        
        # 专业金融词典
        self.financial_lexicon = self._load_financial_lexicon()
            
    def _load_financial_lexicon(self):
        """加载金融情感词典"""
        lexicon = {
            'positive': {
                '超预期', '增长', '利好', '买入', '推荐', '突破', '上涨', '提升', '优秀',
                '强劲', '复苏', '改善', '盈利', '牛市', '金叉', '放量', '涨停', '龙头',
                '价值', '低估', '增持', '强烈推荐', '业绩预增', '订单饱满', '产能释放',
                '新产品发布', '战略合作', '市场份额', '行业景气', '政策利好'
            },
            'negative': {
                '下滑', '亏损', '问询', '减持', '利空', '风险', '下跌', '违规', '下降',
                '疲软', '恶化', '亏损', '利空', '跌停', '死叉', '缩量', '黑天鹅', '暴雷',
                '崩盘', '泡沫', '高估', '减持', '立案', '调查', '警示', '暂停', '退市',
                '业绩预减', '订单下滑', '产能过剩', '竞争加剧', '政策风险', '商誉减值'
            },
            'intensifier': {
                '大幅': 1.8, '显著': 1.5, '明显': 1.3, '大幅增长': 2.0, '大幅下滑': 2.0,
                '急剧': 1.6, '快速': 1.4, '稳步': 1.1, '略有': 0.8, '小幅': 0.7
            }
        }
        return lexicon
    
    def bert_predict(self, texts, batch_size=8):  # 减小batch_size避免内存问题
        """使用 FinBERT 进行情感预测"""
        if not self.use_bert:
            # 如果BERT不可用，返回中性分数
            if not isinstance(texts, list):
                return 0.0
            return [0.0] * len(texts)
        
        if not isinstance(texts, list):
            texts = [texts]
        
        predictions = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            try:
                # Tokenize - 使用慢速tokenizer
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=256,  # 减少最大长度
                    return_tensors="pt"
                ).to(self.device)
                
                # 预测
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)
                    
                # FinBERT 输出: positive, negative, neutral
                for prob in probs:
                    # 计算综合情感得分: positive - negative
                    sentiment_score = float(prob[0] - prob[1])  # positive - negative
                    predictions.append(sentiment_score)
            except Exception as e:
                print(f"BERT预测错误: {e}")
                # 返回中性分数
                predictions.extend([0.0] * len(batch_texts))
        
        return predictions

    def lexicon_enhance(self, text, bert_score):
        """使用金融词典增强情感分析"""
        # 如果没有BERT分数，只使用词典
        if not self.use_bert:
            return self.analyze_text(text)  # 直接使用词典分析
        
        words = set(jieba.lcut(text))
        
        # 词典匹配
        pos_words = words & self.financial_lexicon['positive']
        neg_words = words & self.financial_lexicon['negative']
        
        lexicon_score = len(pos_words) - len(neg_words)
        
        # 程度副词加权
        for word, intensity in self.financial_lexicon['intensifier'].items():
            if word in text:
                lexicon_score *= intensity
        
        # 结合 BERT 分数和词典分数
        if abs(lexicon_score) > 3:  # 词典信号强烈
            combined_score = 0.3 * bert_score + 0.7 * np.sign(bert_score)
        else:
            combined_score = 0.7 * bert_score + 0.3 * (lexicon_score / 10)
        
        return np.clip(combined_score, -1, 1)
    
    def analyze_text(self, text):
        """分析单条文本情感（纯词典方法）"""
        if not isinstance(text, str) or len(text.strip()) == 0:
            return 0.0
        
        # 中文分词
        words = set(jieba.lcut(text))
        
        # 词典匹配
        pos_words = words & self.financial_lexicon['positive']
        neg_words = words & self.financial_lexicon['negative']
        
        lexicon_score = len(pos_words) - len(neg_words)
        
        # 程度副词加权
        for word, intensity in self.financial_lexicon['intensifier'].items():
            if word in text:
                lexicon_score *= intensity
        
        # 归一化到[-1, 1]
        if abs(lexicon_score) > 0:
            normalized_score = np.tanh(lexicon_score / 5)  # 使用tanh平滑
        else:
            normalized_score = 0
        
        return np.clip(normalized_score, -1, 1)
    
    
    def analyze_news_batch(self, news_df):
        """批量分析新闻情感"""
        print(f"开始情感分析，共 {len(news_df)} 条新闻...")
        
        contents = news_df['content'].fillna('').tolist()
        
        if self.use_bert:
            # BERT 基础预测
            print("使用BERT进行情感分析...")
            bert_scores = self.bert_predict(contents, batch_size=4)  # 更小的batch_size
            
            # 词典增强
            enhanced_scores = []
            for i, (text, bert_score) in enumerate(zip(contents, bert_scores)):
                enhanced_score = self.lexicon_enhance(text, bert_score)
                enhanced_scores.append(enhanced_score)
                
                if i % 50 == 0 and i > 0:
                    print(f"  已处理 {i}/{len(contents)} 条新闻")
            
            news_df['sentiment_bert'] = bert_scores
            news_df['sentiment_enhanced'] = enhanced_scores
        else:
            # 只使用词典
            print("使用词典方法进行情感分析...")
            scores = []
            for i, text in enumerate(contents):
                score = self.analyze_text(text)
                scores.append(score)
                
                if i % 100 == 0 and i > 0:
                    print(f"  已处理 {i}/{len(contents)} 条新闻")
            
            news_df['sentiment_score'] = scores
        
        return news_df


# ============================ 3. 多因子引擎 ============================
class AdvancedFactorEngine:
    """高级因子引擎 (技术 + 基本面 + 舆情)"""
    
    def __init__(self):
        self.factors_config = self._load_factors_config()
        
    def _load_factors_config(self):
        """因子配置"""
        return {
            'price': ['open', 'high', 'low', 'close', 'volume', 'turnover'],
            'technical_windows': [5, 10, 20, 30, 60],
            'fundamental': ['pe', 'pb', 'roe', 'gross_margin', 'revenue_growth', 'debt_ratio']
        }
    
    def calculate_technical_factors(self, price_df):
        """计算技术因子"""
        print("计算技术因子...")
        df = price_df.copy()
        
        # 按股票分组计算
        all_factors = []
        
        for code, group in df.groupby('code'):
            group = group.sort_values('date')
            factors = {'date': group['date'].values, 'code': code}
            
            close = group['close'].values
            high = group['high'].values
            low = group['low'].values
            volume = group['volume'].values
            returns = group['returns'].values
            
            # 价格动量类
            for window in self.factors_config['technical_windows']:
                # 收益率动量
                factors[f'ret_{window}d'] = close / np.roll(close, window) - 1
                
                # 波动率
                factors[f'vol_{window}d'] = pd.Series(returns).rolling(window).std().values
                
                # 量价关系
                factors[f'volume_ratio_{window}d'] = volume / pd.Series(volume).rolling(window).mean().values
                
                # 价格位置
                factors[f'position_{window}d'] = (close - pd.Series(close).rolling(window).min().values) / \
                                                 (pd.Series(close).rolling(window).max().values - pd.Series(close).rolling(window).min().values + 1e-8)
            
            # 技术指标
            # RSI
            factors['rsi_14'] = talib.RSI(close, timeperiod=14)
            
            # MACD
            macd, macdsignal, macdhist = talib.MACD(close)
            factors['macd'] = macd
            factors['macd_signal'] = macdsignal
            
            # Bollinger Bands
            upper, middle, lower = talib.BBANDS(close, timeperiod=20)
            factors['bb_width'] = (upper - lower) / middle
            factors['bb_position'] = (close - lower) / (upper - lower)
            
            # ATR
            factors['atr_14'] = talib.ATR(high, low, close, timeperiod=14)
            
            # 异常检测
            factors['price_skew'] = pd.Series(returns).rolling(20).skew().values
            factors['price_kurt'] = pd.Series(returns).rolling(20).kurt().values
            
            all_factors.append(pd.DataFrame(factors))
        
        tech_factors = pd.concat(all_factors, ignore_index=True)
        return tech_factors
    
    def calculate_fundamental_factors(self, fundamental_df, industry_df):
        """计算基本面因子 (行业中性化)"""
        print("计算基本面因子...")
        df = fundamental_df.copy()
        
        # 基础因子
        df['pe_ratio'] = 1 / df['pe']  # 使用倒数
        df['pb_ratio'] = 1 / df['pb']
        df['roe'] = df['roe']
        df['gross_margin'] = df['gross_margin']
        df['revenue_growth'] = df['revenue_growth']
        df['debt_ratio'] = -df['debt_ratio']  # 负债率为负向
        
        # 质量因子
        df['profitability'] = df['roe'] * 0.5 + df['gross_margin'] * 0.5
        df['growth'] = df['revenue_growth']
        
        # 估值因子
        df['value'] = df['pe_ratio'] * 0.5 + df['pb_ratio'] * 0.5
        
        # 合并行业信息
        if industry_df is not None:
            df = pd.merge(df, industry_df, on='code', how='left')
            
            # 行业中性化
            numeric_cols = ['pe_ratio', 'pb_ratio', 'roe', 'gross_margin', 
                           'revenue_growth', 'profitability', 'growth', 'value']
            
            for col in numeric_cols:
                if 'industry' in df.columns:
                    # 计算行业均值
                    industry_mean = df.groupby(['date', 'industry'])[col].transform('mean')
                    # 中性化
                    df[f'{col}_neutral'] = df[col] - industry_mean
        
        return df
    
    def calculate_sentiment_factors(self, news_df, price_df):
        """计算舆情因子"""
        print("计算舆情因子...")
        
        if news_df.empty:
            return pd.DataFrame()
        
        # 确保时间格式
        news_df['date'] = pd.to_datetime(news_df['date'])
        price_df['date'] = pd.to_datetime(price_df['date'])
        
        # 初始化情感分析器
        analyzer = AdvancedSentimentAnalyzer()
        news_df = analyzer.analyze_news_batch(news_df)
        
        sentiment_factors = []
        
        for code, group in price_df.groupby('code'):
            code_news = news_df[news_df['code'] == code].copy()
            
            if code_news.empty:
                continue
            
            for date, date_group in group.groupby('date'):
                # 过去N天的新闻
                mask = (code_news['timestamp'] <= date) & \
                       (code_news['timestamp'] > date - timedelta(days=10))
                
                recent_news = code_news[mask]
                
                if len(recent_news) == 0:
                    continue
                
                # 计算舆情因子
                factors = {
                    'date': date,
                    'code': code,
                    'sentiment_mean': recent_news['sentiment_enhanced'].mean(),
                    'sentiment_std': recent_news['sentiment_enhanced'].std(),
                    'news_count': len(recent_news),
                    'sentiment_pos_ratio': (recent_news['sentiment_enhanced'] > 0.1).mean(),
                    'sentiment_neg_ratio': (recent_news['sentiment_enhanced'] < -0.1).mean(),
                    'sentiment_momentum': recent_news['sentiment_enhanced'].iloc[-1] - \
                                         recent_news['sentiment_enhanced'].iloc[0] if len(recent_news) > 1 else 0
                }
                
                sentiment_factors.append(factors)
        
        if sentiment_factors:
            return pd.DataFrame(sentiment_factors)
        return pd.DataFrame()
    
    def calculate_all_factors(self, price_df, fundamental_df, news_df, industry_df):
        """计算所有因子"""
        # 技术因子
        tech_factors = self.calculate_technical_factors(price_df)
        
        # 基本面因子
        fund_factors = self.calculate_fundamental_factors(fundamental_df, industry_df)
        
        # 舆情因子
        sent_factors = self.calculate_sentiment_factors(news_df, price_df)
        
        # 合并所有因子
        all_factors = pd.merge(tech_factors, fund_factors, on=['date', 'code'], how='left')
        
        if not sent_factors.empty:
            all_factors = pd.merge(all_factors, sent_factors, on=['date', 'code'], how='left')

        # 确保包含close列
        if 'close' not in all_factors.columns:
            close_df = price_df[['date', 'code', 'close']].copy()
            all_factors = pd.merge(all_factors, close_df, on=['date', 'code'], how='left')
        
        # 确保包含returns列
        if 'returns' not in all_factors.columns and 'close' in all_factors.columns:
            all_factors['returns'] = all_factors.groupby('code')['close'].pct_change()

        # 填充缺失值
        all_factors = all_factors.fillna(method='ffill').fillna(0)
        
        print(f"因子计算完成，共 {all_factors.shape[1] - 2} 个因子")
        return all_factors

# ============================ 4. 特征工程模块 ============================
class FeatureEngineer:
    """特征工程 (中性化、标准化、特征选择)"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.quantile_scaler = QuantileTransformer(output_distribution='normal')
        
    def neutralize_features(self, df, factor_cols, industry_col='industry'):
        """行业中性化"""
        if industry_col not in df.columns:
            return df[factor_cols]
        
        neutralized = df[factor_cols].copy()
        
        for col in factor_cols:
            if col in df.columns:
                # 计算行业均值
                industry_mean = df.groupby(['date', industry_col])[col].transform('mean')
                # 中性化
                neutralized[col] = df[col] - industry_mean
        
        return neutralized
    
    def process_features(self, df, factor_cols, industry_col='industry'):
        """特征处理流程"""
        # 1. 行业中性化
        neutralized = self.neutralize_features(df, factor_cols, industry_col)
        
        # 2. 横截面标准化
        processed = neutralized.copy()
        
        for date in df['date'].unique():
            date_mask = df['date'] == date
            if date_mask.sum() > 1:
                # 截面排名
                processed.loc[date_mask] = neutralized.loc[date_mask].rank(pct=True)
        
        # 3. 去极值
        processed = processed.clip(lower=processed.quantile(0.01), 
                                 upper=processed.quantile(0.99), axis=1)
        
        # 4. 标准化
        processed = pd.DataFrame(
            self.scaler.fit_transform(processed),
            columns=processed.columns,
            index=processed.index
        )
        
        return processed

# ============================ 5. 高级AI模型 ============================
class AdvancedStockSelector:
    """高级股票选择器 (LightGBM + 贝叶斯优化)"""
    
    def __init__(self, params=None):
        self.model = None
        self.feature_columns = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.params = params or {
            'n_estimators': 500,
            'max_depth': 8,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
        
    def prepare_data(self, factor_df, forward_days=5, top_pct=0.3):
        """准备训练数据"""
        print(f"准备训练数据，预测未来 {forward_days} 天收益...")
        
        data = factor_df.copy()
        
        # 创建标签：未来N日收益率
        labels = []
        valid_indices = []
        
        for code, group in data.groupby('code'):
            group = group.sort_values('date')
            
            # 计算未来收益
            future_returns = group['close'].shift(-forward_days) / group['close'] - 1
            
            # 计算截面排名
            for idx, row in group.iterrows():
                date = row['date']
                date_data = data[data['date'] == date]
                
                if len(date_data) >= 10:  # 确保有足够股票
                    # 找到当前股票在当天数据中的位置
                    same_date_data = data[data['date'] == date]
                    if not same_date_data.empty:
                        # 找到当前code在当天数据中的行
                        code_mask = same_date_data['code'] == code
                        if code_mask.any():
                            # 计算当天所有股票的未来收益排名
                            same_date_returns = same_date_data['close'].shift(-forward_days) / same_date_data['close'] - 1
                            
                            if not same_date_returns.isna().all():
                                # 获取当前股票的排名
                                code_idx = same_date_data[code_mask].index[0]
                                if code_idx in same_date_returns.index:
                                    rank_value = same_date_returns.rank(pct=True).loc[code_idx]
                                    
                                    if not pd.isna(rank_value):
                                        label = 1 if rank_value > (1 - top_pct) else 0
                                        labels.append(label)
                                        valid_indices.append(idx)
                                    else:
                                        labels.append(0)
                                        valid_indices.append(idx)
                                else:
                                    labels.append(0)
                                    valid_indices.append(idx)
                            else:
                                labels.append(0)
                                valid_indices.append(idx)
                        else:
                            labels.append(0)
                            valid_indices.append(idx)
                    else:
                        labels.append(0)
                        valid_indices.append(idx)
                else:
                    labels.append(0)
                    valid_indices.append(idx)
        
        if valid_indices:
            data = data.loc[valid_indices].copy()
            data['label'] = labels[:len(data)]
        else:
            data['label'] = 0
            print("警告：没有生成有效标签")
        
        print(f"正样本比例: {data['label'].mean():.2%}")
        return data
        

    def bayesian_optimization(self, X_train, y_train, X_val, y_val, n_iter=30):
        """贝叶斯优化调参"""
        
        def lgb_eval(n_estimators, max_depth, learning_rate, subsample, colsample_bytree):
            """LightGBM 评估函数"""
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'n_estimators': int(n_estimators),
                'max_depth': int(max_depth),
                'learning_rate': learning_rate,
                'subsample': max(min(subsample, 1), 0.1),
                'colsample_bytree': max(min(colsample_bytree, 1), 0.1),
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'random_state': RANDOM_SEED,
                'n_jobs': -1,
                'verbose': -1
            }
            
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train)
            
            y_pred = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred)
            return auc
        
        # 定义参数空间
        pbounds = {
            'n_estimators': (100, 1000),
            'max_depth': (3, 12),
            'learning_rate': (0.01, 0.1),
            'subsample': (0.5, 1.0),
            'colsample_bytree': (0.5, 1.0)
        }
        
        # 贝叶斯优化
        optimizer = BayesianOptimization(
            f=lgb_eval,
            pbounds=pbounds,
            random_state=RANDOM_SEED,
            verbose=0
        )
        
        optimizer.maximize(init_points=5, n_iter=n_iter)
        
        # 获取最佳参数
        best_params = optimizer.max['params']
        best_params['n_estimators'] = int(best_params['n_estimators'])
        best_params['max_depth'] = int(best_params['max_depth'])
        
        print(f"最佳参数: {best_params}")
        print(f"最佳AUC: {optimizer.max['target']:.4f}")
        
        return best_params
    
    def train(self, factor_df, feature_cols, val_size=0.2, optimize=True):
        """训练模型"""
        print("开始训练模型...")
        
        # 准备数据
        data = self.prepare_data(factor_df)
        
        # 特征工程
        X = data[feature_cols].copy()
        y = data['label'].values
        
        # 划分时间序列
        dates = data['date'].unique()
        split_idx = int(len(dates) * (1 - val_size))
        train_dates = dates[:split_idx]
        val_dates = dates[split_idx:]
        
        train_mask = data['date'].isin(train_dates)
        val_mask = data['date'].isin(val_dates)
        
        X_train, X_val = X[train_mask], X[val_mask]
        y_train, y_val = y[train_mask], y[val_mask]
        
        # 处理类别不平衡
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=RANDOM_SEED)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        
        print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}")
        print(f"训练集正样本: {y_train.mean():.2%}, 验证集正样本: {y_val.mean():.2%}")
        
        # 贝叶斯优化
        if optimize:
            print("进行贝叶斯优化...")
            best_params = self.bayesian_optimization(X_train, y_train, X_val, y_val, n_iter=20)
            self.params.update(best_params)
        
        # 训练最终模型
        self.model = lgb.LGBMClassifier(
            **self.params,
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
        )
        
        # 特征重要性
        self.feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # 评估
        y_pred = self.model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)
        
        print(f"模型训练完成，验证集AUC: {auc:.4f}")
        print("\nTop 10 重要特征:")
        print(self.feature_importance.head(10))
        
        return self.model
    
    def predict(self, X):
        """预测"""
        if self.model is None:
            raise ValueError("模型未训练")
        
        return self.model.predict_proba(X)[:, 1]

# ============================ 6. 回测引擎 ============================
class AdvancedBacktester:
    """高级回测引擎 - 修复版"""
    
    def __init__(self, initial_cash=1000000, commission=0.0003, slippage=0.0001):
        self.initial_cash = initial_cash
        self.commission = commission
        self.slippage = slippage
        self.strategy_results = {}
        
    def run_backtest(self, signals, price_data, top_n=10, holding_days=5):
        """运行回测"""
        print(f"\n开始回测，初始资金: {self.initial_cash:,.0f}")
        print(f"交易成本: {self.commission:.2%}, 滑点: {self.slippage:.2%}")
        
        # 准备数据
        price_pivot = price_data.pivot(index='date', columns='code', values='close')
        
        # 按信号选股
        portfolio_returns = []
        trade_logs = []
        
        dates = sorted(signals['date'].unique())
        
        for i in range(0, len(dates) - holding_days, holding_days):
            current_date = dates[i]
            
            # 获取今日信号
            today_signals = signals[signals['date'] == current_date]
            
            if len(today_signals) < top_n:
                continue
            
            # 选择 top_n 只股票
            selected_stocks = today_signals.nlargest(top_n, 'pred_prob')
            
            # 计算持有期收益
            for _, stock in selected_stocks.iterrows():
                code = stock['code']
                pred_prob = stock['pred_prob']
                
                # 获取价格
                price_today = price_pivot.loc[current_date, code] if code in price_pivot.columns else None
                price_future = price_pivot.loc[dates[i + holding_days], code] if dates[i + holding_days] in price_pivot.index and code in price_pivot.columns else None
                
                if price_today and price_future and not np.isnan(price_today) and not np.isnan(price_future):
                    # 计算收益 (考虑交易成本)
                    raw_return = (price_future - price_today) / price_today
                    net_return = raw_return - 2 * self.commission - 2 * self.slippage
                    
                    portfolio_returns.append(net_return)
                    
                    trade_logs.append({
                        'date': current_date,
                        'code': code,
                        'buy_price': price_today,
                        'sell_price': price_future,
                        'pred_prob': pred_prob,
                        'raw_return': raw_return,
                        'net_return': net_return,
                        'holding_days': holding_days
                    })
        
        trade_df = pd.DataFrame(trade_logs)
        
        if len(portfolio_returns) == 0:
            print("无有效交易")
            return {}
        
        # 计算绩效指标
        portfolio_returns = np.array(portfolio_returns)
        
        metrics = {
            'total_trades': len(portfolio_returns),
            'win_rate': (portfolio_returns > 0).mean(),
            'avg_return': portfolio_returns.mean(),
            'std_return': portfolio_returns.std(),
            'sharpe_ratio': portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252/holding_days) if portfolio_returns.std() > 0 else 0,
            'max_drawdown': self.calculate_max_drawdown(portfolio_returns),
            'profit_factor': self.calculate_profit_factor(trade_df),
            'avg_holding_days': holding_days
        }
        
        # 年度化收益
        metrics['annual_return'] = (1 + metrics['avg_return']) ** (252 / holding_days) - 1
        
        print("\n" + "="*50)
        print("回测结果:")
        print("="*50)
        print(f"总交易次数: {metrics['total_trades']}")
        print(f"胜率: {metrics['win_rate']:.2%}")
        print(f"平均收益率: {metrics['avg_return']:.2%}")
        print(f"年化收益率: {metrics['annual_return']:.2%}")
        print(f"夏普比率: {metrics['sharpe_ratio']:.2f}")
        print(f"最大回撤: {metrics['max_drawdown']:.2%}")
        print(f"盈亏比: {metrics['profit_factor']:.2f}")
        
        self.strategy_results = metrics
        return metrics, trade_df
    
    def calculate_max_drawdown(self, returns):
        """计算最大回撤"""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def calculate_profit_factor(self, trade_df):
        """计算盈亏比"""
        if trade_df.empty:
            return 0
        gross_profit = trade_df[trade_df['net_return'] > 0]['net_return'].sum()
        gross_loss = abs(trade_df[trade_df['net_return'] < 0]['net_return'].sum())
        return gross_profit / gross_loss if gross_loss != 0 else float('inf')

# ============================ 7. 主执行流程 ============================
def main():
    """主执行函数"""
    print("="*60)
    print("A 股 AI 量化交易系统")
    print("="*60)
    
    # 1. 数据准备
    print("\n[1/6] 获取数据...")
    fetcher = DataFetcher(data_source='simulated')
    
    # 股票池
    universe = [f"{i:06d}.SZ" if i < 300000 else f"{i:06d}.SH" 
               for i in range(1, 51)]  # 50只股票
    
    # 获取数据
    start_date = '2022-01-01'
    end_date = '2023-12-31'
    
    price_data = fetcher.fetch_price_data(start_date, end_date, universe)
    fundamental_data = fetcher.fetch_fundamental_data(price_data['date'].unique(), universe)
    news_data = fetcher.fetch_news_data(start_date, end_date, universe)
    industry_data = fetcher.fetch_industry_data(universe)
    
    print(f"价格数据: {price_data.shape}")
    print(f"基本面数据: {fundamental_data.shape}")
    print(f"新闻数据: {news_data.shape}")
    
    # 2. 因子计算
    print("\n[2/6] 计算因子...")
    factor_engine = AdvancedFactorEngine()
    all_factors = factor_engine.calculate_all_factors(
        price_data, fundamental_data, news_data, industry_data
    )
    
    # 3. 特征工程
    print("\n[3/6] 特征工程...")
    # 选择因子列
    factor_cols = [col for col in all_factors.columns 
                  if col not in ['date', 'code', 'close', 'returns', 'industry'] 
                  and not col.startswith('_')]
    
    feature_engineer = FeatureEngineer()
    processed_features = feature_engineer.process_features(
        all_factors, factor_cols, 'industry'
    )
    
    # 合并回原始数据
    processed_features['date'] = all_factors['date'].values
    processed_features['code'] = all_factors['code'].values
    processed_features['close'] = all_factors['close'].values
    
    # 4. 模型训练
    print("\n[4/6] 训练AI模型...")
    selector = AdvancedStockSelector()
    
    # 划分特征和标签
    feature_cols = [col for col in processed_features.columns 
                   if col not in ['date', 'code', 'close', 'label', 'returns']]
    
    # 训练模型
    model = selector.train(processed_features, feature_cols, optimize=True)
    
    # 5. 生成预测
    print("\n[5/6] 生成预测信号...")
    # 使用所有特征进行预测
    X = processed_features[feature_cols]
    processed_features['pred_prob'] = selector.predict(X)
    
    # 选择信号
    signals = processed_features[['date', 'code', 'pred_prob']].copy()
    signals = signals.dropna()
    
    # 每日选top_n
    daily_signals = []
    for date, group in signals.groupby('date'):
        if len(group) >= 10:
            top_stocks = group.nlargest(10, 'pred_prob')
            daily_signals.append(top_stocks)
    
    signals_df = pd.concat(daily_signals)
    print(f"生成 {len(signals_df)} 条交易信号")
    
    # 6. 回测
    print("\n[6/6] 运行回测...")
    backtester = AdvancedBacktester(
        initial_cash=1000000,
        commission=0.0003,  # 万分之三
        slippage=0.0001     # 万分之一滑点
    )
    
    metrics, trade_df = backtester.run_backtest(
        signals_df, price_data,
        top_n=10,
        holding_days=5
    )
    
    # 7. 特征分析
    print("\n" + "="*50)
    print("特征重要性分析:")
    print("="*50)
    for i, (_, feature_row) in enumerate(selector.feature_importance.head(20).iterrows()):
        print(f"{i+1:2d}. {feature_row['feature']:30s} {feature_row['importance']:.4f}")
    
    # 8. 策略分析
    print("\n" + "="*50)
    print("策略性能分析:")
    print("="*50)
    
    # 按预测概率分组
    signals_df['prob_group'] = pd.qcut(signals_df['pred_prob'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    
    # 合并收益
    merged = pd.merge(signals_df, price_data[['date', 'code', 'returns']], 
                     on=['date', 'code'], how='left')
    
    for group, group_data in merged.groupby('prob_group'):
        avg_return = group_data['returns'].mean() * 100
        hit_rate = (group_data['returns'] > 0).mean() * 100
        count = len(group_data)
        print(f"组 {group}: 样本数={count:4d}, 平均收益={avg_return:6.2f}%, 胜率={hit_rate:5.1f}%")
    
    return {
        'model': selector,
        'factors': all_factors,
        'signals': signals_df,
        'trades': trade_df,
        'metrics': metrics
    }

if __name__ == "__main__":
    # 运行完整系统
    results = main()
