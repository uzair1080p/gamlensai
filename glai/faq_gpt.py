"""
GPT-powered FAQ system for GameLens AI
"""

import os
import json
import pandas as pd
from typing import Dict, Any, Optional, List
from openai import OpenAI
from glai.db import get_db_session
from glai.models import Dataset, ModelVersion, PredictionRun
from glai.predict import load_predictions
from glai.train import load_model_artifacts
import logging

logger = logging.getLogger(__name__)

class GameLensFAQGPT:
    """GPT-powered FAQ system that provides intelligent answers based on actual data"""
    
    def __init__(self):
        self.client = None
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            logger.warning("OpenAI API key not found. FAQ will use fallback answers.")
    
    def generate_context_summary(self, 
                                selected_model: Optional[ModelVersion] = None,
                                selected_dataset: Optional[Dataset] = None,
                                filters: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Generate comprehensive context summary for GPT"""
        
        context = {
            "system_info": {
                "platform": "GameLens AI - ROAS Forecasting Platform",
                "capabilities": [
                    "ROAS prediction from early campaign data (3 days)",
                    "Multi-platform advertising analysis (Unity Ads, Mistplay, etc.)",
                    "Confidence intervals with quantile regression",
                    "Campaign recommendations (Scale, Maintain, Reduce, Cut)",
                    "Feature importance analysis",
                    "Performance validation and metrics"
                ]
            },
            "current_context": {
                "selected_model": None,
                "selected_dataset": None,
                "filters": filters or {}
            },
            "available_data": {
                "datasets": [],
                "models": [],
                "recent_predictions": []
            }
        }
        
        # Get selected model context
        if selected_model:
            context["current_context"]["selected_model"] = {
                "name": selected_model.model_name,
                "version": selected_model.version,
                "target_day": selected_model.target_day,
                "created_at": selected_model.created_at.isoformat() if selected_model.created_at else None,
                "metrics": selected_model.metrics_json or {},
                "feature_importance": {}
            }
            
            # Load feature importance if available
            try:
                artifacts = load_model_artifacts(selected_model.id)
                if artifacts and 'feature_importance' in artifacts:
                    context["current_context"]["selected_model"]["feature_importance"] = artifacts['feature_importance']
            except Exception as e:
                logger.warning(f"Could not load feature importance for model {selected_model.id}: {e}")
        
        # Get selected dataset context
        if selected_dataset:
            context["current_context"]["selected_dataset"] = {
                "name": selected_dataset.canonical_name,
                "platform": selected_dataset.source_platform.value if hasattr(selected_dataset.source_platform, 'value') else str(selected_dataset.source_platform),
                "channel": selected_dataset.channel,
                "game": selected_dataset.game,
                "countries": selected_dataset.countries or [],
                "records": selected_dataset.records,
                "date_range": {
                    "start": selected_dataset.data_start_date.isoformat() if selected_dataset.data_start_date else None,
                    "end": selected_dataset.data_end_date.isoformat() if selected_dataset.data_end_date else None
                },
                "schema_info": {}
            }
            
            # Load dataset schema info
            try:
                if selected_dataset.storage_path and os.path.exists(selected_dataset.storage_path):
                    df = pd.read_parquet(selected_dataset.storage_path)
                    context["current_context"]["selected_dataset"]["schema_info"] = {
                        "columns": list(df.columns),
                        "shape": df.shape,
                        "roas_columns": [col for col in df.columns if col.startswith('roas_d')],
                        "retention_columns": [col for col in df.columns if col.startswith('retention_')],
                        "level_columns": [col for col in df.columns if col.startswith('level_')],
                        "sample_stats": {
                            "avg_cost": df['cost'].mean() if 'cost' in df.columns else None,
                            "avg_revenue": df['revenue'].mean() if 'revenue' in df.columns else None,
                            "avg_installs": df['installs'].mean() if 'installs' in df.columns else None
                        }
                    }
            except Exception as e:
                logger.warning(f"Could not load dataset schema for {selected_dataset.id}: {e}")
        
        # Get available data summary
        try:
            db = get_db_session()
            datasets = db.query(Dataset).filter(Dataset.ingest_completed_at.isnot(None)).limit(10).all()
            models = db.query(ModelVersion).order_by(ModelVersion.created_at.desc()).limit(5).all()
            predictions = db.query(PredictionRun).order_by(PredictionRun.requested_at.desc()).limit(5).all()
            
            context["available_data"]["datasets"] = [
                {
                    "name": d.canonical_name,
                    "platform": d.source_platform.value if hasattr(d.source_platform, 'value') else str(d.source_platform),
                    "records": d.records,
                    "date_range": f"{d.data_start_date} to {d.data_end_date}" if d.data_start_date and d.data_end_date else "Unknown"
                }
                for d in datasets
            ]
            
            context["available_data"]["models"] = [
                {
                    "name": m.model_name,
                    "version": m.version,
                    "target_day": m.target_day,
                    "created_at": m.created_at.isoformat() if m.created_at else None
                }
                for m in models
            ]
            
            context["available_data"]["recent_predictions"] = [
                {
                    "model": p.model_version_id,
                    "dataset": p.dataset_id,
                    "requested_at": p.requested_at.isoformat() if p.requested_at else None,
                    "completed_at": p.completed_at.isoformat() if p.completed_at else None,
                    "n_rows": p.n_rows,
                    "summary": p.summary_json or {}
                }
                for p in predictions
            ]
            
            db.close()
        except Exception as e:
            logger.warning(f"Could not load available data summary: {e}")
        
        return context
    
    def generate_faq_answer(self, 
                           question: str, 
                           context: Dict[str, Any],
                           use_gpt: bool = True) -> str:
        """Generate FAQ answer using GPT or fallback"""
        
        if not use_gpt or not self.client:
            return self._generate_fallback_answer(question, context)
        
        try:
            # Prepare context for GPT
            context_str = json.dumps(context, indent=2)
            
            # Create system prompt
            system_prompt = """You are an expert data scientist and business analyst for GameLens AI, a ROAS (Return on Ad Spend) forecasting platform for mobile game studios.

Your role is to provide intelligent, actionable answers to business questions about advertising campaign performance, ROAS predictions, and optimization strategies.

Key capabilities of the platform:
- Predict ROAS from early campaign data (3 days) using machine learning
- Support multiple advertising platforms (Unity Ads, Mistplay, Facebook Ads, etc.)
- Provide confidence intervals and uncertainty quantification
- Generate campaign recommendations (Scale, Maintain, Reduce, Cut)
- Analyze feature importance and drivers of ROAS
- Support hierarchical data structure: Game > Platform > Channel > Countries

When answering questions:
1. Be specific and actionable
2. Reference the actual data and models available
3. Direct users to the appropriate tabs/features in the platform
4. Provide business context and recommendations
5. Use metrics and data to support your answers
6. If specific data is not available, explain what's needed to answer the question

Always maintain a professional, consultative tone focused on helping game studios optimize their advertising spend."""
            
            # Create user prompt
            user_prompt = f"""Please answer this business question about ROAS forecasting and campaign optimization:

Question: {question}

Context about the current GameLens AI session:
{context_str}

Please provide a comprehensive, actionable answer that helps the user make informed decisions about their advertising campaigns."""
            
            # Call GPT
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=800,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"GPT FAQ generation failed: {e}")
            return self._generate_fallback_answer(question, context)
    
    def _generate_fallback_answer(self, question: str, context: Dict[str, Any]) -> str:
        """Generate fallback answer when GPT is not available"""
        
        question_lower = question.lower()
        
        # Check if we have model and dataset context
        has_model = context.get("current_context", {}).get("selected_model")
        has_dataset = context.get("current_context", {}).get("selected_dataset")
        has_ai_recommendations = context.get("has_predictions", False) and context.get("ai_recommendations", {}).get("count", 0) > 0
        
        if "roi" in question_lower and ("100%" in question or "d15" in question_lower or "d30" in question_lower or "d90" in question_lower):
            if has_model and has_dataset:
                return f"To determine when ROI reaches 100% on this channel, run predictions using model '{has_model['name']}' on dataset '{has_dataset['name']}' in the Predictions tab. The system will show projected ROAS over time with confidence intervals to identify when break-even is achieved."
            elif has_ai_recommendations:
                return "Based on the current AI recommendations, ROI analysis requires historical ROAS data over time. The current recommendations show campaign-level actions (Scale/Maintain/Reduce/Cut) but don't include time-series ROAS projections. For 100% ROI timing, you would need to train a model with historical data that includes ROAS progression over D15, D30, D90 periods."
            else:
                return "Please select a model and dataset to get ROI projections. Go to the Model Training tab to train or select a model, then run predictions to see when 100% ROI will be achieved."
        
        elif "continue" in question_lower and ("campaign" in question_lower or "pause" in question_lower):
            if has_model and has_dataset:
                return f"Campaign continuation recommendations are available in the Predictions tab using model '{has_model['name']}'. The system categorizes campaigns as Scale, Maintain, Reduce, or Cut based on predicted ROAS and confidence intervals."
            elif has_ai_recommendations:
                ai_summary = context.get("ai_recommendations", {})
                actions = ai_summary.get("actions_breakdown", {})
                action_text = ", ".join([f"{count} {action}" for action, count in actions.items() if count > 0])
                return f"Based on current AI analysis: {action_text}. Review the specific recommendations above for each campaign. 'Cut' campaigns should be paused, 'Reduce' campaigns need optimization, 'Maintain' campaigns can continue as-is, and 'Scale' campaigns should receive increased budget."
            else:
                return "Please select a model and dataset to get campaign recommendations. Use the Model Training tab to make your selections, then check the Predictions tab for specific guidance."
        
        elif "cpi" in question_lower and ("profitability" in question_lower or "d30" in question_lower or "d90" in question_lower):
            if has_model and has_dataset:
                return f"Target CPI for profitability can be calculated using model '{has_model['name']}' on dataset '{has_dataset['name']}'. Run predictions in the Predictions tab to see the relationship between CPI and projected ROAS, helping identify the maximum CPI for profitable campaigns."
            else:
                return "Please select a model and dataset to analyze CPI requirements. Use the Model Training tab to make your selections, then run predictions to see CPI vs ROAS relationships."
        
        elif "retention rate" in question_lower and ("d7" in question_lower or "d15" in question_lower or "d30" in question_lower):
            if has_model and has_dataset:
                return f"Required retention rates for D30 ROAS goals can be analyzed using model '{has_model['name']}' on dataset '{has_dataset['name']}'. Check the Model Training tab for feature importance to see how retention impacts ROAS predictions."
            else:
                return "Please select a model and dataset to analyze retention requirements. Use the Model Training tab to train a model and view feature importance, showing how retention rates affect ROAS predictions."
        
        elif "scale aggressively" in question_lower or "ready to scale" in question_lower:
            if has_model and has_dataset:
                return f"Scaling recommendations are available in the Predictions tab using model '{has_model['name']}'. Look for campaigns with high predicted ROAS and narrow confidence intervals as candidates for aggressive scaling."
            else:
                return "Please select a model and dataset to get scaling recommendations. Use the Model Training tab to make your selections, then check the Predictions tab for specific guidance."
        
        elif "cut immediately" in question_lower or "campaigns should we cut" in question_lower:
            if has_model and has_dataset:
                return f"Campaign cutting recommendations are available in the Predictions tab using model '{has_model['name']}'. Look for campaigns with low predicted ROAS and wide confidence intervals as candidates for immediate cuts."
            else:
                return "Please select a model and dataset to get cutting recommendations. Use the Model Training tab to make your selections, then check the Predictions tab for specific guidance."
        
        else:
            return "I can help you with questions about model performance, campaign recommendations, ROAS predictions, and data insights. Please select a model and dataset for more specific answers, or ask a more specific question."


def get_faq_gpt() -> GameLensFAQGPT:
    """Get singleton FAQ GPT instance"""
    if not hasattr(get_faq_gpt, '_instance'):
        get_faq_gpt._instance = GameLensFAQGPT()
    return get_faq_gpt._instance
