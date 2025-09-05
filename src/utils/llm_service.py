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
                openai.api_key = self.api_key
                self.available = True
            except Exception as e:
                logger.error(f"Error setting up OpenAI API: {e}")
                self.available = False
    
    def is_available(self) -> bool:
        """Check if LLM service is available"""
        return self.available
    
    def answer_faq_question(self, question: str, context_data: Dict, faq_content: str = "") -> str:
        """Answer FAQ question using GPT with context from the dashboard"""
        
        if not self.available or not OPENAI_AVAILABLE:
            return "❌ LLM service not available. Please configure OpenAI API key in .env file."
        
        try:
            # Prepare context from dashboard data
            context_prompt = self._build_context_prompt(context_data, faq_content)
            
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
            
            user_prompt = f"""Context from current dashboard:
{context_prompt}

FAQ Content (if available):
{faq_content}

Question: {question}

Please provide a helpful answer based on the context and FAQ content above."""

            # Make API call
            response = openai.ChatCompletion.create(
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
    
    def _build_context_prompt(self, context_data: Dict, faq_content: str) -> str:
        """Build context prompt from dashboard data"""
        
        context_parts = []
        
        # Add model performance metrics
        if 'metrics' in context_data:
            metrics = context_data['metrics']
            context_parts.append("Current Model Performance:")
            for metric, value in metrics.items():
                context_parts.append(f"- {metric}: {value}")
        
        # Add prediction insights
        if 'predictions' in context_data:
            predictions = context_data['predictions']
            context_parts.append(f"\nPrediction Insights:")
            context_parts.append(f"- Total predictions made: {len(predictions)}")
            if 'roas_prediction' in predictions.columns:
                avg_pred = predictions['roas_prediction'].mean()
                context_parts.append(f"- Average predicted ROAS: {avg_pred:.3f}")
        
        # Add feature importance
        if 'top_features' in context_data:
            features = context_data['top_features']
            context_parts.append(f"\nTop Important Features:")
            for i, (feature, importance) in enumerate(features[:5], 1):
                context_parts.append(f"{i}. {feature}: {importance:.3f}")
        
        # Add platform insights
        if 'best_platform' in context_data:
            platform = context_data['best_platform']
            context_parts.append(f"\nPlatform Performance:")
            context_parts.append(f"- Best performing platform: {platform}")
        
        # Add recommendations summary
        if 'recommendations' in context_data:
            recs = context_data['recommendations']
            context_parts.append(f"\nRecommendations Summary:")
            if not recs.empty and 'recommendation' in recs.columns:
                rec_counts = recs['recommendation'].value_counts()
                for rec, count in rec_counts.items():
                    context_parts.append(f"- {rec}: {count} campaigns")
        
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

            response = openai.ChatCompletion.create(
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
