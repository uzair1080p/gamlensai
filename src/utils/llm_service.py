import os
from typing import Dict, List, Optional
import logging

# Try to import OpenAI with error handling
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None

# Try to load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, will use system environment variables

logger = logging.getLogger(__name__)

class GameLensLLMService:
    """LLM service for GameLens AI - handles FAQ answering with GPT"""
    
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.model = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
        self.max_tokens = int(os.getenv('OPENAI_MAX_TOKENS', '500'))
        self.temperature = float(os.getenv('OPENAI_TEMPERATURE', '0.7'))
        
        if not OPENAI_AVAILABLE:
            logger.warning("OpenAI library not available. FAQ functionality will be limited.")
            self.available = False
        elif not self.api_key:
            logger.warning("OpenAI API key not found. FAQ functionality will be limited.")
            self.available = False
        else:
            try:
                # Initialize OpenAI client with new API format
                self.client = openai.OpenAI(api_key=self.api_key)
                self.available = True
            except Exception as e:
                logger.error(f"Error setting up OpenAI API: {e}")
                self.available = False
    
    def is_available(self) -> bool:
        """Check if LLM service is available"""
        return self.available
    
    def answer_faq_question(self, question: str, context_data: Dict, faq_content: str = "") -> str:
        """Answer FAQ question using GPT with context from the dashboard.

        Adds simple intent detection to tailor the prompt and focuses the
        answers on concrete metrics rather than generic guidance.
        """
        
        if not self.available or not OPENAI_AVAILABLE:
            return "❌ LLM service not available. Please configure OpenAI API key in .env file."
        
        try:
            # Prepare context and intent-specific guidance
            context_prompt = self._build_context_prompt(context_data, faq_content)
            intent = self._detect_intent(question)
            intent_prompt = self._build_intent_prompt(intent, context_data)
            
            # Create the full prompt
            system_prompt = """You are an AI assistant for GameLens AI, a ROAS (Return on Ad Spend) forecasting platform for mobile game advertising. 
            
Your role is to answer questions about the platform, data analysis, model performance, and provide insights based on the current dashboard data.

Guidelines:
- Be concise and professional
- Use specific numbers and metrics when available
- Provide actionable insights when possible
- If you don't have enough information, say so clearly
- Focus on ROAS forecasting, mobile advertising, and data analytics topics
"""
            
            user_prompt = f"""Context from current dashboard (JSON-like summaries first, then narrative):
{context_prompt}

Intent-specific guidance:
{intent_prompt}

FAQ Content from your documentation (if available):
{faq_content}

Question: {question}

Please provide a helpful answer based on the context and FAQ content above. If the FAQ content contains relevant information, use it to provide accurate, detailed answers. If the question is about your specific data or model performance, use the dashboard context to provide data-driven insights."""

            # Make API call using new OpenAI API format
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return f"❌ Error generating answer: {str(e)}"

    def _detect_intent(self, question: str) -> str:
        """Lightweight intent classification for better prompts."""
        q = question.lower()
        if any(k in q for k in ["100% roi", "when will roi", "break-even", "payback", "d15", "d30", "d90"]):
            return "roi_timeline"
        if any(k in q for k in ["continue", "pause", "scale", "cut", "reduce", "maintain", "budget"]):
            return "budget_action"
        if any(k in q for k in ["projected roas", "keep spending", "same pace", "projection"]):
            return "projection_same_spend"
        if any(k in q for k in ["level progression", "highest quality", "long-term", "retention", "cohort"]):
            return "level_quality_channels"
        if any(k in q for k in ["compare", "actual", "ai forecast", "day"]):
            return "forecast_vs_actual"
        return "general_insight"

    def _build_intent_prompt(self, intent: str, ctx: Dict) -> str:
        """Return concise, numbers-first guidance for the LLM per intent."""
        metrics = ctx.get('metrics', {}) or {}
        pred_summary = ctx.get('pred_summary', {}) or {}
        top_features = ctx.get('top_features', []) or []
        scope = ctx.get('scope', {}) or {}
        level_summary = ctx.get('level_progression_summary', {}) or {}

        head = f"INTENT: {intent}\nSCOPE: {scope}\n"

        if intent == "roi_timeline":
            return head + (
                "Goal: Estimate the earliest D-day where expected cumulative ROAS ≈ 1.0.\n"
                f"Use prediction summary: {pred_summary} and any ROAS-by-day if available.\n"
                f"Model metrics: {metrics}. Output a concrete day with a short rationale."
            )
        if intent == "budget_action":
            return head + (
                f"Goal: Action (scale/maintain/reduce/pause). Use metrics {metrics}, drivers {top_features}, predictions {pred_summary}.\n"
                "Output 3 bullets: recommendation, confidence (high/med/low), and watchouts."
            )
        if intent == "projection_same_spend":
            return head + (
                f"Goal: Project ROAS assuming current spend pace. Use {pred_summary} and {metrics}.\n"
                "Provide D7/D30 style outlook and risks."
            )
        if intent == "level_quality_channels":
            return head + (
                f"Goal: Rank channels/platforms by player quality via level progression summary: {level_summary}.\n"
                "Prefer median max level, drop-off, and link back to ROAS differences."
            )
        if intent == "forecast_vs_actual":
            return head + (
                f"Goal: Compare actuals vs forecast succinctly. Use metrics {metrics}.\n"
                "Report absolute/percentage errors and confidence coverage if present."
            )
        return head + "Goal: Provide focused insights using the numbers above (2-4 bullets)."
    
    def _build_context_prompt(self, context_data: Dict, faq_content: str) -> str:
        """Build comprehensive context prompt from dashboard data"""
        
        context_parts = []
        
        # Add model performance metrics with detailed analysis
        if 'metrics' in context_data:
            metrics = context_data['metrics']
            context_parts.append("📊 CURRENT MODEL PERFORMANCE:")
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    if metric == 'r2':
                        context_parts.append(f"- R² Score: {value:.4f} ({'Excellent' if value > 0.8 else 'Good' if value > 0.6 else 'Fair' if value > 0.4 else 'Poor'} model fit)")
                    elif metric == 'rmse':
                        context_parts.append(f"- RMSE: {value:.4f} ({'Low' if value < 0.1 else 'Moderate' if value < 0.3 else 'High'} prediction error)")
                    elif metric == 'mape':
                        context_parts.append(f"- MAPE: {value:.4f} ({'Excellent' if value < 0.1 else 'Good' if value < 0.2 else 'Fair' if value < 0.3 else 'Poor'} accuracy)")
                    elif metric == 'mae':
                        context_parts.append(f"- MAE: {value:.4f} (Mean Absolute Error)")
                    elif metric == 'confidence_coverage':
                        context_parts.append(f"- Confidence Coverage: {value:.4f} ({'Good' if value > 0.8 else 'Fair' if value > 0.6 else 'Poor'} confidence interval accuracy)")
                    else:
                        context_parts.append(f"- {metric}: {value:.4f}")
                else:
                    context_parts.append(f"- {metric}: {value}")
        
        # Add detailed prediction insights
        if 'predictions' in context_data:
            predictions = context_data['predictions']
            context_parts.append(f"\n🎯 PREDICTION INSIGHTS:")
            context_parts.append(f"- Total predictions made: {len(predictions):,}")
            
            if hasattr(predictions, 'columns') and 'roas_prediction' in predictions.columns:
                avg_pred = predictions['roas_prediction'].mean()
                min_pred = predictions['roas_prediction'].min()
                max_pred = predictions['roas_prediction'].max()
                std_pred = predictions['roas_prediction'].std()
                
                context_parts.append(f"- Average predicted ROAS: {avg_pred:.3f}")
                context_parts.append(f"- ROAS range: {min_pred:.3f} to {max_pred:.3f}")
                context_parts.append(f"- ROAS standard deviation: {std_pred:.3f}")
                
                # ROAS distribution analysis
                high_roas = (predictions['roas_prediction'] > 1.0).sum()
                medium_roas = ((predictions['roas_prediction'] > 0.5) & (predictions['roas_prediction'] <= 1.0)).sum()
                low_roas = (predictions['roas_prediction'] <= 0.5).sum()
                
                context_parts.append(f"- High ROAS campaigns (>1.0): {high_roas:,} ({high_roas/len(predictions)*100:.1f}%)")
                context_parts.append(f"- Medium ROAS campaigns (0.5-1.0): {medium_roas:,} ({medium_roas/len(predictions)*100:.1f}%)")
                context_parts.append(f"- Low ROAS campaigns (≤0.5): {low_roas:,} ({low_roas/len(predictions)*100:.1f}%)")
            
            # Confidence interval analysis
            if hasattr(predictions, 'columns') and 'roas_lower_bound' in predictions.columns and 'roas_upper_bound' in predictions.columns:
                avg_confidence_width = (predictions['roas_upper_bound'] - predictions['roas_lower_bound']).mean()
                context_parts.append(f"- Average confidence interval width: {avg_confidence_width:.3f}")
        
        # Add feature importance with business interpretation
        if 'top_features' in context_data:
            features = context_data['top_features']
            context_parts.append(f"\n🔍 TOP PREDICTIVE FEATURES:")
            for i, (feature, importance) in enumerate(features[:10], 1):
                # Add business interpretation for common features
                interpretation = ""
                if 'retention' in feature.lower():
                    interpretation = " (User engagement indicator)"
                elif 'roas' in feature.lower():
                    interpretation = " (Historical performance)"
                elif 'cost' in feature.lower() or 'cpi' in feature.lower():
                    interpretation = " (Acquisition efficiency)"
                elif 'revenue' in feature.lower():
                    interpretation = " (Monetization potential)"
                elif 'level' in feature.lower():
                    interpretation = " (Game progression)"
                elif 'platform' in feature.lower():
                    interpretation = " (Channel performance)"
                
                context_parts.append(f"{i}. {feature}: {importance:.3f}{interpretation}")
        
        # Add platform insights
        if 'best_platform' in context_data:
            platform = context_data['best_platform']
            context_parts.append(f"\n📱 PLATFORM PERFORMANCE:")
            context_parts.append(f"- Best performing platform: {platform}")
        
        # Add recommendations summary with actionable insights
        if 'recommendations' in context_data:
            recs = context_data['recommendations']
            context_parts.append(f"\n💡 RECOMMENDATIONS SUMMARY:")
            if hasattr(recs, 'empty') and not recs.empty:
                if 'roas_improvement' in recs.columns:
                    avg_improvement = recs['roas_improvement'].mean()
                    max_improvement = recs['roas_improvement'].max()
                    context_parts.append(f"- Average ROAS improvement potential: {avg_improvement:.3f}")
                    context_parts.append(f"- Maximum ROAS improvement potential: {max_improvement:.3f}")
                
                if 'roas_prediction' in recs.columns:
                    avg_rec_roas = recs['roas_prediction'].mean()
                    context_parts.append(f"- Average recommended campaign ROAS: {avg_rec_roas:.3f}")
                
                context_parts.append(f"- Total recommendations generated: {len(recs)}")
            else:
                context_parts.append("- No specific recommendations available")
        
        # Add data quality insights
        if 'data_quality' in context_data:
            quality = context_data['data_quality']
            context_parts.append(f"\n📈 DATA QUALITY:")
            for key, value in quality.items():
                context_parts.append(f"- {key}: {value}")
        
        # Add model training insights
        if 'training_info' in context_data:
            training = context_data['training_info']
            context_parts.append(f"\n🤖 MODEL TRAINING:")
            for key, value in training.items():
                context_parts.append(f"- {key}: {value}")
        
        return "\n".join(context_parts) if context_parts else "No specific context data available."
    
    def generate_insights(self, data_summary: Dict) -> str:
        """Generate general insights about the data and model"""
        
        if not self.available or not OPENAI_AVAILABLE:
            return "❌ LLM service not available for insights generation."
        
        try:
            system_prompt = """You are an AI data analyst for GameLens AI. Generate concise, actionable insights about the ROAS forecasting model and data."""
            
            user_prompt = f"""Based on this data summary, provide 3-5 key insights:

{data_summary}

Focus on:
- Model performance trends
- Data quality observations  
- Actionable recommendations
- Potential improvements

Keep insights concise and business-focused."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return f"❌ Error generating insights: {str(e)}"
