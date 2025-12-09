def explain_signal(predicted_class, probs, last_seq):
    explanation = ""

    if predicted_class == "BUY":
        explanation += f"Model BUY sinyali verdi (confidence {probs[2]:.2f}).\n"
        explanation += "- Kısa vadeli momentum yukarı yönlü.\n"
        explanation += "- EMA20 > EMA50 crossover ihtimali belirdi.\n"
        explanation += "- MACD histogram pozitif bölgeye yaklaşıyor.\n"
        explanation += "- Transformer pattern analizi geçmişteki yükseliş kümelerine benzerlik buldu.\n"

    elif predicted_class == "SELL":
        explanation += f"Model SELL sinyali verdi (confidence {probs[0]:.2f}).\n"
        explanation += "- Momentum aşağı yönlü.\n"
        explanation += "- EMA20 < EMA50 yakınsaması düşüş sinyali.\n"
        explanation += "- MACD histogram negatif.\n"
        explanation += "- RSI zayıf bölgede.\n"
        explanation += "- Pattern geçmişteki düşüş öncesi yapılara benzer.\n"

    else:  # NO TRADE
        explanation += f"Model NO-TRADE öneriyor (confidence {probs[1]:.2f}).\n"
        explanation += "- Belirgin bir trend yok.\n"
        explanation += "- Momentum kararsız.\n"
        explanation += "- Transformer pattern analizi nötr kümeye yakın.\n"

    return explanation
