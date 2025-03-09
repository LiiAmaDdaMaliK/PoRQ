from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional, Dict
import uvicorn
import numpy as np
import librosa
import librosa.display
import io
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from difflib import SequenceMatcher

app = FastAPI()

# تحميل نموذج التعرف على الصوت
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

# نصوص مرجعية لمقارنة الأداء (يمكن توسيعها لاحقًا)
REFERENCE_TEXTS = {
    "quran": "الحمد لله رب العالمين الرحمن الرحيم مالك يوم الدين",
    "speech": "This is a sample speech for testing purposes."
}

# نموذج البيانات لتقييم الجودة
class EvaluationRequest(BaseModel):
    user_id: str
    performance_data: str  # يمكن أن يكون نصًا أو بيانات رقمية
    category: str  # نوع التقييم (تلاوة، محتوى، أداء صوتي...)

def calculate_similarity(reference: str, transcript: str) -> float:
    """حساب نسبة التطابق بين النص المرجعي والتلاوة المنطوقة."""
    return SequenceMatcher(None, reference, transcript).ratio() * 100

def classify_performance(score: float) -> str:
    """تصنيف الأداء بناءً على درجة التشابه."""
    if score >= 90:
        return "ممتاز"
    elif score >= 75:
        return "جيد جدًا"
    elif score >= 60:
        return "جيد"
    elif score >= 40:
        return "مقبول"
    else:
        return "ضعيف"

# تخزين مؤقت للبيانات
users_scores = {}
users_rewards = {}

# توزيع المكافآت الرباعي
REWARD_DISTRIBUTION = {
    "gold": 0.25,  # 25% ذهب رقمي (PAXG)
    "stablecoin": 0.25,  # 25% عملات مستقرة (USDC / DAI)
    "bitcoin": 0.25,  # 25% بيتكوين (BTC)
    "project_token": 0.25  # 25% توكن خاص بالمشروع
}

def calculate_rewards(score: float) -> Dict[str, float]:
    """حساب قيمة المكافآت بناءً على الأداء."""
    base_reward = score / 10  # تحويل النسبة إلى مكافأة مالية رمزية
    return {asset: base_reward * percentage for asset, percentage in REWARD_DISTRIBUTION.items()}

@app.post("/evaluate")
def evaluate_performance(request: EvaluationRequest):
    """تقييم أداء المستخدم ومنح المكافآت بناءً على الربط الرباعي."""
    reference_text = REFERENCE_TEXTS.get(request.category, "")
    score = calculate_similarity(reference_text, request.performance_data)
    classification = classify_performance(score)
    rewards = calculate_rewards(score)
    
    if request.user_id not in users_scores:
        users_scores[request.user_id] = []
        users_rewards[request.user_id] = {asset: 0 for asset in REWARD_DISTRIBUTION}
    
    users_scores[request.user_id].append(score)
    
    # تحديث رصيد المكافآت
    for asset, amount in rewards.items():
        users_rewards[request.user_id][asset] += amount
    
    return {
        "user_id": request.user_id,
        "score": score,
        "classification": classification,
        "category": request.category,
        "rewards": rewards
    }

@app.get("/user_rewards/{user_id}")
def get_user_rewards(user_id: str):
    """إرجاع المكافآت المجمعة للمستخدم بناءً على الأصول الأربعة."""
    if user_id in users_rewards:
        return {"user_id": user_id, "rewards": users_rewards[user_id]}
    return {"error": "User not found"}

@app.post("/withdraw")
def withdraw_rewards(user_id: str, asset: str, amount: float):
    """السحب من المكافآت بناءً على الأصول المتاحة."""
    if user_id not in users_rewards:
        return {"error": "User not found"}
    if asset not in users_rewards[user_id]:
        return {"error": "Invalid asset"}
    if users_rewards[user_id][asset] < amount:
        return {"error": "Insufficient balance"}
    
    users_rewards[user_id][asset] -= amount
    return {"user_id": user_id, "withdrawn": {asset: amount}, "remaining_balance": users_rewards[user_id]}

@app.get("/leaderboard")
def get_leaderboard():
    """إرجاع قائمة بأفضل المستخدمين أداءً مع مكافآتهم."""
    sorted_users = sorted(users_scores.items(), key=lambda x: np.mean(x[1]), reverse=True)
    return {
        "leaderboard": [
            {
                "user_id": u,
                "avg_score": np.mean(s),
                "classification": classify_performance(np.mean(s)),
                "rewards": users_rewards[u]
            } for u, s in sorted_users
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

