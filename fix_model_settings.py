#!/usr/bin/env python3
"""
Script to fix the model settings that are causing poor Gemini responses
"""

def show_fix_commands():
    """Show commands to fix the model settings"""
    
    print("🔧 Fixing Model Settings for Better Gemini Responses")
    print("=" * 60)
    
    print("\n❌ Current Issues Found:")
    print("🔧 Gemini Config: model=gemini-2.0-flash, temp=1.0, max_tokens=1")
    print("🔧 System Prompt: string...")
    print("")
    print("Problems:")
    print("1. max_tokens=1 → Should be 300 (allows only 1 word responses)")
    print("2. temp=1.0 → Should be 0.8 (too random)")
    print("3. System prompt is 'string' → Should be proper instructions")
    
    print("\n✅ Fix Commands:")
    print("=" * 30)
    
    print("\n1. Fix Model Settings:")
    print("""
curl -X PUT "http://localhost:8000/api/v1/models/{your-model-id}" \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer YOUR_TOKEN" \\
  -d '{
    "temperature": 80,
    "max_tokens": 300,
    "system_prompt": "You are a helpful AI assistant for phone calls. Provide clear, conversational responses that are easy to understand when spoken. Be friendly and professional. Give complete answers, not just single words. If you don'\''t understand something, ask for clarification. Keep responses between 1-3 sentences for good voice interaction. Be helpful and try to answer questions thoroughly."
  }'
""")
    
    print("\n2. Fix Agent System Prompt:")
    print("""
curl -X PUT "http://localhost:8000/api/v1/agent/{your-agent-id}" \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer YOUR_TOKEN" \\
  -d '{
    "system_prompt": "You are a helpful AI assistant for phone calls. Provide clear, conversational responses that are easy to understand when spoken. Be friendly and professional. Give complete answers, not just single words. If you don'\''t understand something, ask for clarification. Keep responses between 1-3 sentences for good voice interaction. Be helpful and try to answer questions thoroughly."
  }'
""")
    
    print("\n3. Get Your Model ID:")
    print("""
curl -X GET "http://localhost:8000/api/v1/gemini/models/gemini" \\
  -H "Authorization: Bearer YOUR_TOKEN"
""")
    
    print("\n4. Get Your Agent ID:")
    print("""
curl -X GET "http://localhost:8000/api/v1/agent" \\
  -H "Authorization: Bearer YOUR_TOKEN"
""")
    
    print("\n5. Test the Fix:")
    print("""
curl -X POST "http://localhost:8000/api/v1/gemini/test-text-generation" \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer YOUR_TOKEN" \\
  -d '{
    "model_id": "your-model-id",
    "prompt": "How are you?",
    "temperature": 0.8,
    "max_tokens": 300
  }'
""")
    
    print("\n📊 Expected Results After Fix:")
    print("Instead of: 'I'")
    print("You should get: 'I'\''m doing well, thank you for asking! How can I help you today?'")
    
    print("\n🔍 Debug Information:")
    print("After the fix, you should see:")
    print("🔧 Gemini Config: model=gemini-2.0-flash, temp=0.8, max_tokens=300")
    print("🔧 System Prompt: You are a helpful AI assistant for phone calls...")
    print("🔧 User Prompt: How are you?")
    print("✅ Gemini generated response: I'\''m doing well, thank you for asking! How can I help you today?")

def show_quick_fix():
    """Show a quick fix for immediate testing"""
    
    print("\n🚀 Quick Fix for Immediate Testing:")
    print("=" * 40)
    
    print("\nIf you want to test immediately without API calls:")
    print("1. Go to your database")
    print("2. Find the model table")
    print("3. Update the record with:")
    print("   - temperature = 80")
    print("   - max_tokens = 300")
    print("   - system_prompt = 'You are a helpful AI assistant for phone calls. Provide clear, conversational responses that are easy to understand when spoken. Be friendly and professional. Give complete answers, not just single words.'")
    
    print("\nSQL Command (if you have database access):")
    print("""
UPDATE model 
SET 
    temperature = 80,
    max_tokens = 300,
    system_prompt = 'You are a helpful AI assistant for phone calls. Provide clear, conversational responses that are easy to understand when spoken. Be friendly and professional. Give complete answers, not just single words. If you don'\''t understand something, ask for clarification. Keep responses between 1-3 sentences for good voice interaction. Be helpful and try to answer questions thoroughly.'
WHERE model_name LIKE '%gemini%';
""")

def main():
    """Main function"""
    show_fix_commands()
    show_quick_fix()
    
    print("\n🎯 Summary:")
    print("The issue is that your model has max_tokens=1, which only allows 1-word responses.")
    print("Fix the model settings and you'll get proper responses!")
    
    print("\n⚡ Quick Test:")
    print("After fixing, test with: 'How are you?'")
    print("Expected: 'I'm doing well, thank you for asking! How can I help you today?'")
    print("Instead of: 'I'")

if __name__ == "__main__":
    main()
