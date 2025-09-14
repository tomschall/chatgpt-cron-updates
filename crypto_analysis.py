#!/usr/bin/env python3
import os
import json
import math
import time
import requests
import pandas as pd
from tabulate import tabulate
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple
import numpy as np

COINGECKO_BASE = "https://api.coingecko.com/api/v3"

def fetch_top_cryptos(limit=20):
    """Holt die Top Kryptowährungen von CoinGecko"""
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": limit,
        "page": 1,
        "price_change_percentage": "1h,24h,7d,30d",
        "sparkline": "false",
    }
    
    headers = {
        "User-Agent": "CryptoAnalysis/1.0"
    }
    
    try:
        response = requests.get(
            f"{COINGECKO_BASE}/coins/markets",
            params=params,
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Fehler beim Abrufen der Daten: {e}")
        return []

def calculate_technical_indicators(df):
    """Berechnet technische Indikatoren für die Analyse"""
    # Volatilität berechnen
    df['volatility_24h'] = df['price_change_percentage_24h_in_currency'].abs()
    df['volatility_7d'] = df['price_change_percentage_7d_in_currency'].abs()
    df['volatility_30d'] = df['price_change_percentage_30d_in_currency'].abs()
    
    # Momentum Score (kombiniert verschiedene Zeiträume)
    df['momentum_score'] = (
        df['price_change_percentage_24h_in_currency'] * 0.4 +
        df['price_change_percentage_7d_in_currency'] * 0.3 +
        df['price_change_percentage_30d_in_currency'] * 0.3
    )
    
    # Relative Stärke Index (vereinfacht)
    df['rsi_approximation'] = 50 + (df['momentum_score'] / 2)
    df['rsi_approximation'] = df['rsi_approximation'].clip(0, 100)
    
    # Trend Score
    df['trend_score'] = 0
    df.loc[df['price_change_percentage_24h_in_currency'] > 0, 'trend_score'] += 1
    df.loc[df['price_change_percentage_7d_in_currency'] > 0, 'trend_score'] += 1
    df.loc[df['price_change_percentage_30d_in_currency'] > 0, 'trend_score'] += 1
    
    # Volumen-zu-Marktkapitalisierung Ratio
    df['volume_mcap_ratio'] = df['total_volume'] / df['market_cap']
    
    return df

def predict_next_moves(df):
    """Erstellt Vorhersagen für die nächsten Moves"""
    predictions = []
    
    for _, coin in df.iterrows():
        # Basis-Vorhersage basierend auf technischen Indikatoren
        momentum = coin['momentum_score']
        volatility = coin['volatility_24h']
        trend = coin['trend_score']
        rsi = coin['rsi_approximation']
        volume_ratio = coin['volume_mcap_ratio']
        
        # Wahrscheinlichkeiten berechnen
        prob_up = 0.5  # Basis-Wahrscheinlichkeit
        prob_down = 0.5
        prob_sideways = 0.3
        
        # Momentum-basierte Anpassungen
        if momentum > 5:
            prob_up += 0.2
            prob_down -= 0.1
        elif momentum < -5:
            prob_down += 0.2
            prob_up -= 0.1
        
        # RSI-basierte Anpassungen
        if rsi > 70:
            prob_down += 0.15
            prob_up -= 0.1
        elif rsi < 30:
            prob_up += 0.15
            prob_down -= 0.1
        
        # Trend-basierte Anpassungen
        if trend >= 2:
            prob_up += 0.1
        elif trend <= 1:
            prob_down += 0.1
        
        # Volatilität berücksichtigen
        if volatility > 10:
            prob_sideways += 0.2
            prob_up *= 0.8
            prob_down *= 0.8
        
        # Volumen-Ratio berücksichtigen
        if volume_ratio > 0.1:  # Hohes Volumen
            prob_up += 0.05
            prob_down += 0.05
        
        # Normalisierung
        total = prob_up + prob_down + prob_sideways
        prob_up /= total
        prob_down /= total
        prob_sideways /= total
        
        # Erwartete Preisänderung
        expected_change = momentum * 0.5  # Konservative Schätzung
        
        # Risiko-Score
        risk_score = min(volatility / 20, 1.0)  # 0-1 Skala
        
        predictions.append({
            'symbol': coin['symbol'],
            'name': coin['name'],
            'current_price': coin['current_price'],
            'market_cap': coin['market_cap'],
            'prob_up': prob_up,
            'prob_down': prob_down,
            'prob_sideways': prob_sideways,
            'expected_change_pct': expected_change,
            'risk_score': risk_score,
            'confidence': 1 - risk_score,
            'recommendation': get_recommendation(prob_up, prob_down, prob_sideways, risk_score)
        })
    
    return predictions

def get_recommendation(prob_up, prob_down, prob_sideways, risk_score):
    """Generiert Empfehlung basierend auf Wahrscheinlichkeiten"""
    if prob_up > 0.6 and risk_score < 0.5:
        return "STRONG BUY"
    elif prob_up > 0.55 and risk_score < 0.6:
        return "BUY"
    elif prob_down > 0.6 and risk_score < 0.5:
        return "STRONG SELL"
    elif prob_down > 0.55 and risk_score < 0.6:
        return "SELL"
    elif prob_sideways > 0.5:
        return "HOLD"
    elif prob_up > prob_down:
        return "WEAK BUY"
    else:
        return "WEAK SELL"

def generate_report(predictions):
    """Generiert einen umfassenden Report"""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    
    report = f"""# Krypto-Analyse & Prognose Report
**Stand: {timestamp}**

## Executive Summary

Dieser Report analysiert die 20 größten Kryptowährungen nach Marktkapitalisierung und bietet Vorhersagen für die nächsten Bewegungen basierend auf technischen Indikatoren, Momentum-Analyse und Marktfundamenten.

## Top 20 Kryptowährungen - Detaillierte Analyse

"""
    
    # Sortiere nach Marktkapitalisierung
    predictions_sorted = sorted(predictions, key=lambda x: x['market_cap'], reverse=True)
    
    # Erstelle Tabelle
    table_data = []
    for i, pred in enumerate(predictions_sorted, 1):
        table_data.append([
            i,
            pred['symbol'],
            pred['name'],
            f"${pred['current_price']:,.2f}",
            f"${pred['market_cap']:,.0f}",
            f"{pred['prob_up']:.1%}",
            f"{pred['prob_down']:.1%}",
            f"{pred['prob_sideways']:.1%}",
            f"{pred['expected_change_pct']:+.1f}%",
            pred['recommendation'],
            f"{pred['confidence']:.1%}"
        ])
    
    headers = [
        "Rank", "Symbol", "Name", "Preis", "Market Cap", 
        "Wahrscheinlichkeit ↑", "Wahrscheinlichkeit ↓", "Seitwärts", 
        "Erwartete Änderung", "Empfehlung", "Vertrauen"
    ]
    
    report += tabulate(table_data, headers=headers, tablefmt="github")
    
    # Zusätzliche Analysen
    report += f"""

## Marktübersicht

### Stärkste Momentum-Kandidaten
"""
    
    # Top 5 nach Momentum
    momentum_sorted = sorted(predictions, key=lambda x: x['expected_change_pct'], reverse=True)[:5]
    for pred in momentum_sorted:
        report += f"- **{pred['symbol']}**: {pred['expected_change_pct']:+.1f}% erwartete Änderung (Vertrauen: {pred['confidence']:.1%})\n"
    
    report += f"""

### Niedrigste Risiko-Kandidaten
"""
    
    # Top 5 nach niedrigstem Risiko
    low_risk = sorted(predictions, key=lambda x: x['risk_score'])[:5]
    for pred in low_risk:
        report += f"- **{pred['symbol']}**: Risiko-Score {pred['risk_score']:.2f} (Vertrauen: {pred['confidence']:.1%})\n"
    
    report += f"""

### Kaufempfehlungen (Strong Buy/Buy)
"""
    
    buy_recommendations = [p for p in predictions if p['recommendation'] in ['STRONG BUY', 'BUY']]
    if buy_recommendations:
        for pred in buy_recommendations:
            report += f"- **{pred['symbol']}** ({pred['name']}): {pred['recommendation']} - {pred['prob_up']:.1%} Wahrscheinlichkeit für Aufwärtsbewegung\n"
    else:
        report += "- Keine starken Kaufempfehlungen im aktuellen Markt\n"
    
    report += f"""

## Methodik & Hinweise

### Verwendete Indikatoren:
- **Momentum Score**: Kombination aus 24h, 7d und 30d Preisänderungen
- **RSI-Approximation**: Relative Stärke basierend auf Momentum
- **Trend Score**: Anzahl positiver Zeiträume
- **Volatilität**: Absolute Preisänderungen
- **Volumen-Ratio**: Verhältnis von Volumen zu Marktkapitalisierung

### Risikobewertung:
- **Niedrig (0.0-0.3)**: Stabile, etablierte Coins
- **Mittel (0.3-0.7)**: Moderate Volatilität
- **Hoch (0.7-1.0)**: Sehr volatile, spekulative Coins

### Empfehlungen:
- **STRONG BUY/SELL**: Hohe Wahrscheinlichkeit (>60%) bei niedrigem Risiko
- **BUY/SELL**: Moderate Wahrscheinlichkeit (>55%) bei akzeptablem Risiko
- **HOLD**: Seitwärtsbewegung wahrscheinlich
- **WEAK BUY/SELL**: Leichte Tendenz, aber unsicher

## Wichtige Hinweise

⚠️ **Risikowarnung**: Diese Analyse dient nur zu Informationszwecken und stellt keine Finanzberatung dar. Kryptowährungen sind hochvolatil und mit erheblichen Risiken verbunden.

📊 **Datenquelle**: CoinGecko API
🕒 **Aktualität**: {timestamp}
🔄 **Empfehlung**: Regelmäßige Überwachung der Marktentwicklung

---
*Generiert von Crypto Analysis Tool v1.0*
"""
    
    return report

def main():
    print("🔄 Lade Top 20 Kryptowährungen...")
    data = fetch_top_cryptos(20)
    
    if not data:
        print("❌ Fehler beim Laden der Daten")
        return
    
    print(f"✅ {len(data)} Kryptowährungen geladen")
    
    # DataFrame erstellen
    df = pd.DataFrame(data)
    
    # Technische Indikatoren berechnen
    print("📊 Berechne technische Indikatoren...")
    df = calculate_technical_indicators(df)
    
    # Vorhersagen erstellen
    print("🔮 Erstelle Vorhersagen...")
    predictions = predict_next_moves(df)
    
    # Report generieren
    print("📝 Generiere Report...")
    report = generate_report(predictions)
    
    # Report speichern
    with open("crypto_analysis_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    
    # JSON für weitere Verarbeitung
    with open("crypto_predictions.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    
    print("✅ Report gespeichert als 'crypto_analysis_report.md'")
    print("✅ Vorhersagen gespeichert als 'crypto_predictions.json'")
    
    # Kurze Zusammenfassung ausgeben
    print("\n" + "="*60)
    print("KURZE ZUSAMMENFASSUNG")
    print("="*60)
    
    buy_signals = [p for p in predictions if p['recommendation'] in ['STRONG BUY', 'BUY']]
    sell_signals = [p for p in predictions if p['recommendation'] in ['STRONG SELL', 'SELL']]
    
    print(f"🟢 Kaufempfehlungen: {len(buy_signals)}")
    for signal in buy_signals[:3]:  # Top 3
        print(f"   - {signal['symbol']}: {signal['recommendation']} ({signal['prob_up']:.1%})")
    
    print(f"🔴 Verkaufsempfehlungen: {len(sell_signals)}")
    for signal in sell_signals[:3]:  # Top 3
        print(f"   - {signal['symbol']}: {signal['recommendation']} ({signal['prob_down']:.1%})")
    
    avg_confidence = np.mean([p['confidence'] for p in predictions])
    print(f"📊 Durchschnittliches Vertrauen: {avg_confidence:.1%}")

if __name__ == "__main__":
    main()
