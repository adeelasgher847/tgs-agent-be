#!/usr/bin/env python3
"""
Script to help improve agent responses by updating system prompts and model settings
"""

def show_improvements():
    """Show how to improve agent responses"""
    
    print("🚀 Improving Agent Responses for Better Gemini Output")
    print("=" * 60)
    
    print("\n📋 Issue Identified:")
    print("Gemini responded with just 'This' to 'Whose name is once there?'")
    print("This is too short and not helpful for voice interactions.")
    
    print("\n🔧 Improvements Made:")
    print("1. ✅ Increased max_tokens from 200 to 300")
    print("2. ✅ Increased temperature from 0.7 to 0.8 for more natural responses")
    print("3. ✅ Improved system prompt with better instructions")
    print("4. ✅ Added debugging logs to see what's sent to Gemini")
    
    print("\n📝 Better System Prompt:")
    print("""
You are a helpful AI assistant for phone calls. 
- Provide clear, conversational responses that are easy to understand when spoken
- Be friendly and professional
- Give complete answers, not just single words
- If you don't understand something, ask for clarification
- Keep responses between 1-3 sentences for good voice interaction
- Be helpful and try to answer questions thoroughly
""")
    
    print("\n🔧 How to Update Your Agent's System Prompt:")
    print("""
curl -X PUT "http://localhost:8000/api/v1/agent/{agent_id}" \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer YOUR_TOKEN" \\
  -d '{
    "system_prompt": "You are a helpful AI assistant for phone calls. Provide clear, conversational responses that are easy to understand when spoken. Be friendly and professional. Give complete answers, not just single words. If you don'\''t understand something, ask for clarification. Keep responses between 1-3 sentences for good voice interaction. Be helpful and try to answer questions thoroughly."
  }'
""")
    
    print("\n🔧 How to Update Your Model Settings:")
    print("""
curl -X PUT "http://localhost:8000/api/v1/models/{model_id}" \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer YOUR_TOKEN" \\
  -d '{
    "temperature": 80,
    "max_tokens": 300
  }'
""")
    
    print("\n🧪 Test the Improvements:")
    print("""
# Test with the same question
curl -X POST "http://localhost:8000/api/v1/gemini/test-text-generation" \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer YOUR_TOKEN" \\
  -d '{
    "model_id": "your-model-id",
    "prompt": "Whose name is once there?",
    "system_prompt": "You are a helpful AI assistant for phone calls. Provide clear, conversational responses that are easy to understand when spoken. Be friendly and professional. Give complete answers, not just single words.",
    "temperature": 0.8,
    "max_tokens": 300
  }'
""")
    
    print("\n📊 Expected Better Response:")
    print("Instead of: 'This'")
    print("You should get: 'I'\''m not sure I understand your question completely. Could you please clarify what you'\''re asking about? Are you referring to a specific person, place, or situation?'")
    
    print("\n🔍 Debug Information:")
    print("The server will now log:")
    print("- 🔧 Gemini Config: model, temperature, max_tokens")
    print("- 🔧 System Prompt: first 100 characters")
    print("- 🔧 User Prompt: what the user said")
    print("- ✅ Gemini generated response: the full response")
    
    print("\n💡 Additional Tips:")
    print("1. Make sure your Gemini API key is valid and not expired")
    print("2. Consider using a more specific system prompt for your use case")
    print("3. Test different temperature values (0.7-0.9) for different response styles")
    print("4. Adjust max_tokens based on how long you want responses to be")

def main():
    """Main function"""
    show_improvements()
    
    print("\n🎯 Next Steps:")
    print("1. Update your agent's system prompt")
    print("2. Update your model's temperature and max_tokens")
    print("3. Test with a voice call")
    print("4. Check the server logs for debugging info")
    print("5. Adjust settings based on the results")

if __name__ == "__main__":
    main()
