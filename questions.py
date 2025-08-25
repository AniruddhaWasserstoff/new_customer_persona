import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
from pathlib import Path

load_dotenv()

# Add the Q&A folder to the path to import both scripts
sys.path.append(os.path.join(os.path.dirname(__file__), 'Q&A'))

try:
    from webfill import SmartWebScraper
except ImportError:
    print("Warning: webfill.py not found in Q&A folder. Website auto-fill will not work.")
    SmartWebScraper = None

try:
    from gformfill import MarketingAnalyzer
except ImportError:
    print("Warning: gformfill.py not found in Q&A folder. CSV auto-fill will not work.")
    MarketingAnalyzer = None

load_dotenv()

class QuestionnaireConfig:
    """Configuration class for the questionnaire"""
    def __init__(self, **kwargs):
        # Auto-fill settings
        self.auto_fill_mode = kwargs.get('auto_fill_mode', False)
        self.fill_source = kwargs.get('fill_source', 'manual')  # 'website', 'csv', 'manual'
        
        # Data source settings
        self.website_url = kwargs.get('website_url', '')
        self.csv_file_path = kwargs.get('csv_file_path', '')
        self.domain_hint = kwargs.get('domain_hint', '')
        
        # Business settings
        self.business_type = kwargs.get('business_type', '')
        self.business_type_description = kwargs.get('business_type_description', '')
        
        # Processing settings
        self.max_pages = kwargs.get('max_pages', 15)
        self.max_workers = kwargs.get('max_workers', 2)
        self.openai_api_key = kwargs.get('openai_api_key', os.getenv('OPENAI_API_KEY'))
        
        # Pipeline settings
        self.interactive_mode = kwargs.get('interactive_mode', True)
        self.save_to_file = kwargs.get('save_to_file', True)
        self.progress_callback = kwargs.get('progress_callback', None)
        
        # Pre-filled answers (for pipeline use)
        self.pre_filled_answers = kwargs.get('pre_filled_answers', {})

class EnhancedSalesPageQuestionnaire:
    def __init__(self, config: Optional[QuestionnaireConfig] = None):
        self.config = config or QuestionnaireConfig()
        self.responses = {}
        self.business_type = self.config.business_type
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.openai_client = None
        self.website_data = None
        self.csv_data = None
        self.auto_fill_mode = self.config.auto_fill_mode
        self.fill_source = self.config.fill_source
        
        # Initialize OpenAI client
        if self.config.openai_api_key:
            self.openai_client = OpenAI(api_key=self.config.openai_api_key)
            self._log("✅ OpenAI client initialized")
        else:
            self._log("⚠️ No OpenAI API key found. Auto-fill will not work.")
    
    def _log(self, message: str):
        """Log messages - can be captured by pipeline"""
        if self.config.progress_callback:
            self.config.progress_callback(message)
        elif self.config.interactive_mode:
            print(message)
    
    def clear_screen(self):
        """Clear the terminal screen for better UX"""
        if self.config.interactive_mode:
            os.system('cls' if os.name == 'nt' else 'clear')
        
    def display_banner(self):
        """Display the program banner"""
        if not self.config.interactive_mode:
            return
            
        print("=" * 80)
        print("🚀 ENHANCED SALES PAGE QUESTIONNAIRE")
        print("🎯 Built to adapt to any business model with maximum output & minimal input")
        print("✨ Now with AI-powered auto-fill from website analysis & CSV data!")
        print("=" * 80)
        print()
    
    def get_business_type(self):
        """Get the business type from user or config"""
        if self.business_type:
            self._log(f"✅ Business Type: {self.business_type}")
            self.responses["business_type"] = self.business_type
            if self.config.business_type_description:
                self.responses["business_type_description"] = self.config.business_type_description
            return
        
        if not self.config.interactive_mode:
            raise ValueError("Business type must be provided in non-interactive mode")
        
        self.clear_screen()
        print("🌳 STEP 1: BUSINESS TYPE IDENTIFICATION")
        print("-" * 50)
        print("What type of product/service do you offer?")
        print()
        
        business_types = {
            "1": ("Physical Product", "DTC, eCommerce"),
            "2": ("Digital Product", "course, template, etc."),
            "3": ("SaaS", "monthly/annual subscription app"),
            "4": ("Service", "freelancer, consultant, agency"),
            "5": ("B2B/Enterprise", "tool or solution"),
            "6": ("Marketplace/Platform", "connecting users"),
            "7": ("Other", "explain your business type")
        }
        
        for key, (name, desc) in business_types.items():
            print(f"{key}. {name} ({desc})")
        
        print()
        while True:
            choice = input("Enter your choice (1-7): ").strip()
            if choice in business_types:
                business_name, _ = business_types[choice]
                self.business_type = business_name
                self.responses["business_type"] = business_name
                
                if choice == "7":  # Other
                    other_desc = input("Please explain your business type: ").strip()
                    self.responses["business_type_description"] = other_desc
                    self.business_type = f"Other - {other_desc}"
                
                print(f"\n✅ Business Type Selected: {self.business_type}")
                input("\nPress Enter to continue...")
                break
            else:
                print("❌ Invalid choice. Please select 1-7.")
    
    def ask_auto_fill_option(self):
        """Ask if user wants to auto-fill or use pre-configured option"""
        if not self.config.interactive_mode:
            # Use pre-configured settings
            if self.config.auto_fill_mode:
                if self.config.fill_source == "website":
                    return self.setup_website_analysis()
                elif self.config.fill_source == "csv":
                    return self.setup_csv_analysis()
            return True
        
        self.clear_screen()
        print("🤖 STEP 2: AUTO-FILL OPTIONS")
        print("-" * 50)
        print(f"Business Type: {self.business_type}")
        print()
        print("How would you like to fill the questionnaire?")
        print()
        print("1. 🌐 Auto-fill from website (AI-powered)")
        print("2. 📊 Auto-fill from CSV/Excel file (Google Forms, surveys, etc.)")
        print("3. ✏️ Manual entry (traditional)")
        print()
        
        while True:
            choice = input("Enter your choice (1-3): ").strip()
            
            if choice == "1":
                return self.handle_website_option()
            elif choice == "2":
                return self.handle_csv_option()
            elif choice == "3":
                print("✅ Manual entry mode selected.")
                self.fill_source = "manual"
                self.responses["data_source"] = {"type": "manual"}
                input("Press Enter to continue...")
                return True
            else:
                print("❌ Invalid choice. Please select 1-3.")
    
    def handle_website_option(self):
        """Handle website auto-fill option"""
        if not self.openai_client or not SmartWebScraper:
            self._log("❌ Website auto-fill not available. Missing dependencies.")
            return False
        
        website_url = self.config.website_url
        if not website_url and self.config.interactive_mode:
            website_url = input("\nEnter your website URL: ").strip()
        
        if not website_url:
            self._log("❌ Website URL is required for auto-fill.")
            return False
        
        return self.setup_website_analysis(website_url)
    
    def handle_csv_option(self):
        """Handle CSV auto-fill option"""
        if not self.openai_client or not MarketingAnalyzer:
            self._log("❌ CSV auto-fill not available. Missing dependencies.")
            return False
        
        csv_file = self.config.csv_file_path
        if not csv_file and self.config.interactive_mode:
            csv_file = input("\nEnter path to your CSV/Excel file: ").strip()
        
        if not csv_file or not Path(csv_file).exists():
            self._log("❌ Valid CSV file path is required for auto-fill.")
            return False
        
        domain_hint = self.config.domain_hint
        if not domain_hint and self.config.interactive_mode:
            domain_hint = input("Website/domain name (optional): ").strip() or None
        
        return self.setup_csv_analysis(csv_file, domain_hint)
    
    def setup_website_analysis(self, website_url: str = None):
        """Setup website analysis"""
        url = website_url or self.config.website_url
        if not url.startswith(('http://', 'https://')):
            url = f"https://{url}"
        
        self.responses["data_source"] = {
            "type": "website",
            "url": url
        }
        self.auto_fill_mode = True
        self.fill_source = "website"
        
        # Run website analysis
        if self.analyze_website(url):
            self._log("✅ Website analysis complete! Ready for auto-fill.")
            return True
        else:
            self._log("❌ Website analysis failed.")
            return False
    
    def setup_csv_analysis(self, csv_file: str = None, domain_hint: str = None):
        """Setup CSV analysis"""
        file_path = csv_file or self.config.csv_file_path
        hint = domain_hint or self.config.domain_hint
        
        self.responses["data_source"] = {
            "type": "csv",
            "file_path": file_path,
            "domain_hint": hint
        }
        self.auto_fill_mode = True
        self.fill_source = "csv"
        
        # Run CSV analysis
        if self.analyze_csv(file_path, hint):
            self._log("✅ CSV analysis complete! Ready for auto-fill.")
            return True
        else:
            self._log("❌ CSV analysis failed.")
            return False
    
    def analyze_website(self, website_url):
        """Analyze website using the webfill.py scraper"""
        self._log(f"🔍 Analyzing website: {website_url}")
        self._log("This may take a few minutes...")
        
        try:
            # Initialize the smart web scraper
            scraper = SmartWebScraper(
                base_url=website_url,
                openai_api_key=self.config.openai_api_key,
                max_workers=self.config.max_workers,
                max_pages=self.config.max_pages
            )
            
            # Run the scraper
            results = scraper.run()
            
            if results and results.get('marketing_analysis'):
                self.website_data = {
                    'final_summary': results.get('final_summary', ''),
                    'marketing_analysis': results.get('marketing_analysis', ''),
                    'page_summaries': results.get('summaries', []),
                    'stats': results.get('stats', {})
                }
                
                self._log(f"✅ Successfully analyzed {len(results.get('summaries', []))} pages")
                return True
            else:
                self._log("❌ No data could be extracted from the website")
                return False
                
        except Exception as e:
            self._log(f"❌ Error analyzing website: {str(e)}")
            return False
    
    def analyze_csv(self, csv_file, domain_hint=None):
        """Analyze CSV file using the gformfill.py analyzer"""
        self._log(f"📊 Analyzing CSV file: {csv_file}")
        self._log("Processing data and generating marketing analysis...")
        
        try:
            # Initialize the marketing analyzer
            analyzer = MarketingAnalyzer(openai_api_key=self.config.openai_api_key)
            
            # Load and analyze the data
            df = analyzer.load_data(csv_file)
            content, columns_used = analyzer.prepare_content_for_analysis(df)
            
            if not content.strip():
                self._log("❌ No analyzable content found in the file")
                return False
            
            # Generate marketing analysis
            analysis, detected_domain = analyzer.generate_marketing_analysis(content, domain_hint)
            
            # Store the results
            self.csv_data = {
                'marketing_analysis': analysis,
                'detected_domain': detected_domain,
                'raw_content': content,
                'columns_used': columns_used,
                'rows_processed': df.shape[0],
                'file_name': Path(csv_file).name
            }
            
            self._log(f"✅ Successfully analyzed {df.shape[0]} rows from {len(columns_used)} columns")
            self._log(f"🎯 Detected domain: {detected_domain}")
            return True
            
        except Exception as e:
            self._log(f"❌ Error analyzing CSV: {str(e)}")
            return False
    
    def auto_fill_with_ai(self, questions_data):
        """Use AI to auto-fill questions based on website or CSV data"""
        if not self.openai_client:
            return questions_data
        
        self._log(f"🤖 AI Auto-filling questionnaire using {self.fill_source} data...")
        
        # Prepare context based on data source
        if self.fill_source == "website" and self.website_data:
            context = f"""
Website Analysis Data:
{self.website_data.get('marketing_analysis', '')}

Additional Context:
{self.website_data.get('final_summary', '')}

Business Type: {self.business_type}
Data Source: Website Analysis
"""
        elif self.fill_source == "csv" and self.csv_data:
            context = f"""
CSV/Form Analysis Data:
{self.csv_data.get('marketing_analysis', '')}

Raw Content Sample:
{self.csv_data.get('raw_content', '')[:1000]}...

Business Type: {self.business_type}
Data Source: CSV/Excel Analysis
Detected Domain: {self.csv_data.get('detected_domain', 'Unknown')}
"""
        else:
            self._log("❌ No analysis data available for auto-fill")
            return questions_data
        
        filled_data = {}
        
        for section_name, questions in questions_data.items():
            filled_data[section_name] = {}
            
            for question_id, question_info in questions.items():
                if isinstance(question_info, dict) and "question" in question_info:
                    question_text = question_info["question"]
                    hint = question_info.get("hint", "")
                else:
                    question_text = question_info
                    hint = ""
                
                # Check for pre-filled answers first
                pre_filled_key = f"{section_name}.{question_id}"
                if pre_filled_key in self.config.pre_filled_answers:
                    ai_answer = self.config.pre_filled_answers[pre_filled_key]
                    self._log(f"✓ Pre-filled: {question_text[:50]}...")
                else:
                    # Get AI response for this question
                    ai_answer = self.get_ai_answer(question_text, hint, context)
                    self._log(f"✓ AI Filled: {question_text[:50]}...")
                
                filled_data[section_name][question_id] = {
                    "question": question_text,
                    "answer": ai_answer,
                    "auto_filled": True,
                    "source": self.fill_source
                }
        
        return filled_data
    
    def get_ai_answer(self, question, hint, context):
        """Get AI answer for a specific question"""
        try:
            prompt = f"""
Based on the analysis data provided, answer the following business questionnaire question.

Question: {question}
{f"Hint: {hint}" if hint else ""}

Context from analysis:
{context}

Instructions:
- Provide a direct, specific answer based on the analysis data
- If the information is not clearly available in the data, respond with "Not found in analysis"
- Keep answers concise but informative
- Use actual phrases and information from the analysis when possible
- Don't make assumptions or create information not present in the data
- Focus on extracting relevant details that directly answer the question

Answer:"""

            response = self.openai_client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {"role": "system", "content": "You are a business analyst helping to fill out a sales page questionnaire based on marketing analysis data. Be precise and only use information that's clearly present in the provided data."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.1
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Clean up the answer
            if not answer or len(answer) < 5:
                return "Not found in analysis"
            
            return answer
            
        except Exception as e:
            self._log(f"⚠️ AI answer failed for question: {str(e)}")
            return "Not found in analysis"
    
    def process_questions_section(self, section_name: str, questions_dict: Dict, mandatory: bool = True):
        """Process a section of questions - can be called programmatically or interactively"""
        if self.auto_fill_mode:
            # Auto-fill with AI
            filled_data = self.auto_fill_with_ai({section_name: questions_dict})[section_name]
            self.responses[section_name] = filled_data
            
            if self.config.interactive_mode:
                # Allow user to review and edit
                self._log(f"🤖 Questions auto-filled using {self.fill_source} data! Please review and edit if needed:")
                print()
                
                for i, (q_id, q_data) in enumerate(self.responses[section_name].items(), 1):
                    print(f"Question {i}/{len(questions_dict)}:")
                    print(f"📋 {q_data['question']}")
                    print(f"🤖 AI Answer ({q_data.get('source', 'unknown')}): {q_data['answer']}")
                    print()
                    
                    if mandatory and not q_data['answer'].strip():
                        edit = input("This question is mandatory. Please provide an answer: ").strip()
                        while not edit:
                            edit = input("Answer required: ").strip()
                    else:
                        edit = input("Press Enter to keep, type new answer, or 'skip' to leave empty: ").strip()
                    
                    if edit and edit.lower() != 'skip':
                        self.responses[section_name][q_id]["answer"] = edit
                        self.responses[section_name][q_id]["auto_filled"] = False
                    elif edit.lower() == 'skip':
                        self.responses[section_name][q_id]["answer"] = ""
                    
                    print("✅ Answer confirmed!\n")
        else:
            # Manual entry or non-interactive mode
            if not self.config.interactive_mode:
                # For pipeline use, create empty responses that can be filled later
                self.responses[section_name] = {}
                for q_id, q_info in questions_dict.items():
                    question_text = q_info["question"] if isinstance(q_info, dict) else q_info
                    self.responses[section_name][q_id] = {
                        "question": question_text,
                        "answer": "",
                        "auto_filled": False,
                        "source": "pipeline"
                    }
                return
            
            # Interactive manual entry
            self.responses[section_name] = {}
            
            for i, (q_id, q_info) in enumerate(questions_dict.items(), 1):
                question_text = q_info["question"] if isinstance(q_info, dict) else q_info
                hint = q_info.get("hint", "") if isinstance(q_info, dict) else ""
                
                print(f"Question {i}/{len(questions_dict)}:")
                print(f"📋 {question_text}")
                if hint:
                    print(f"💡 Hint: {hint}")
                print()
                
                if mandatory:
                    answer = input("Your answer: ").strip()
                    while not answer:
                        print("❌ This question is mandatory. Please provide an answer.")
                        answer = input("Your answer: ").strip()
                else:
                    answer = input("Your answer (optional, press Enter to skip): ").strip()
                
                self.responses[section_name][q_id] = {
                    "question": question_text,
                    "answer": answer,
                    "auto_filled": False,
                    "source": "manual"
                }
                
                if answer:
                    print("✅ Answer recorded!\n")
                else:
                    print("⏭️ Skipped.\n")
    
    def ask_core_questions(self):
        """Ask the 5 mandatory core questions for all business types"""
        if self.config.interactive_mode:
            self.clear_screen()
            print("🔀 STEP 3: CORE QUESTIONS (Mandatory for ALL Business Types)")
            print("-" * 60)
            print(f"Business Type: {self.business_type}")
            if self.auto_fill_mode:
                print(f"🤖 Auto-fill mode: ON ({self.fill_source.upper()})")
            print()
        
        core_questions = {
            "what_you_sell": {
                "question": "What do you sell, in one sentence?",
                "hint": "Skip fancy words. Be punchy. E.g., 'We help freelancers get paid faster with auto-invoicing.'"
            },
            "painful_problem": {
                "question": "What big, painful problem does this solve?",
                "hint": "What's frustrating the customer right before they find you?"
            },
            "transformation_result": {
                "question": "What's the big transformation/result your customer experiences?",
                "hint": "Life after buying. What changes emotionally + practically?"
            },
            "trust_credibility": {
                "question": "Why should someone trust you?",
                "hint": "Proof, credibility, social proof, experience, success stories?"
            },
            "irresistible_offer": {
                "question": "What's the irresistible offer (and price)?",
                "hint": "What do they get? Bonuses? Guarantees? Any urgency?"
            }
        }
        
        self.process_questions_section("core_questions", core_questions, mandatory=True)
        
        if self.config.interactive_mode:
            input("Press Enter to continue to business-specific questions...")
    
    def ask_business_specific_questions(self):
        """Ask questions specific to the business type"""
        if self.config.interactive_mode:
            self.clear_screen()
            print(f"🌿 STEP 4: {self.business_type.upper()} SPECIFIC QUESTIONS")
            print("-" * 60)
        
        questions = self.get_business_specific_questions()
        
        if not questions:
            self._log("No specific questions for this business type.")
            return
        
        # Convert questions to the format expected by process_questions_section
        questions_dict = {}
        for i, question in enumerate(questions, 1):
            questions_dict[f"specific_q{i}"] = {
                "question": question,
                "hint": ""
            }
        
        self.process_questions_section("business_specific_questions", questions_dict, mandatory=False)
        
        if self.config.interactive_mode:
            input("Press Enter to continue...")
    
    def get_business_specific_questions(self):
        """Return specific questions based on business type"""
        questions_map = {
            "Physical Product": [
                "What makes your product different from competitors?",
                "Do you have customer reviews or user-generated content?",
                "Is there a sensory experience (feel, smell, design) that sets it apart?",
                "Can you bundle this with anything for more perceived value?"
            ],
            "Digital Product": [
                "What's the #1 takeaway/result people will get from this course/template?",
                "How long will it take for them to see results?",
                "Have students/customers used it with success? Examples?",
                "Is there a community, support, or bonus?"
            ],
            "SaaS": [
                "What pain point does your tool solve daily/weekly?",
                "What does your onboarding look like?",
                "What's your 'Aha Moment' inside the product?",
                "What objections do people have before signing up? (List 2-3)"
            ],
            "Service": [
                "What results do you deliver and how fast?",
                "Why should they choose *you* over any other freelancer/agency?",
                "What does the process look like? (Timeline, deliverables, meetings?)",
                "What's one client success story that makes you proud?"
            ],
            "B2B/Enterprise": [
                "What business metrics do you impact? (Revenue, churn, CSAT, etc.)",
                "Who are your typical decision-makers?",
                "Do you integrate with any existing systems/tools?",
                "What's the ROI or cost-saving case study you like to brag about?"
            ],
            "Marketplace/Platform": [
                "Who are the two sides of your marketplace?",
                "What value does each side get from joining?",
                "How do you create trust between users (reviews, verification, support)?",
                "What happens in the first 5 minutes after someone signs up?"
            ]
        }
        
        return questions_map.get(self.business_type, [])
    
    def ask_objection_handler_questions(self):
        """Ask optional objection handler questions"""
        if self.config.interactive_mode:
            self.clear_screen()
            print("⚠️ STEP 5: OBJECTION HANDLER SECTION (Optional but Powerful)")
            print("-" * 60)
        
        if not self.auto_fill_mode and self.config.interactive_mode:
            proceed = input("Would you like to answer objection handler questions? (y/n): ").strip().lower()
            if proceed not in ['y', 'yes']:
                self._log("⏭️ Skipping objection handler questions.")
                return
        
        objection_questions = {
            "objection_q1": {
                "question": "What's the biggest hesitation people have before buying/signing up?",
                "hint": ""
            },
            "objection_q2": {
                "question": "How do you destroy that hesitation with logic or proof?",
                "hint": ""
            }
        }
        
        self.process_questions_section("objection_handler", objection_questions, mandatory=False)
        
        if self.config.interactive_mode:
            input("\nPress Enter to continue to final questions...")
    
    def ask_emotional_copy_questions(self):
        """Ask the final 3 emotional/copy gold questions"""
        if self.config.interactive_mode:
            self.clear_screen()
            print("🔥 STEP 6: EMOTIONAL/COPY GOLD SECTION (Final 3 Questions)")
            print("-" * 60)
        
        emotional_questions = {
            "emotional_q1": {
                "question": "What's the one belief your product proves wrong?",
                "hint": ""
            },
            "emotional_q2": {
                "question": "What are your customers secretly dreaming about when they buy this?",
                "hint": ""
            },
            "emotional_q3": {
                "question": "If they don't act now—what will they regret or miss?",
                "hint": ""
            }
        }
        
        self.process_questions_section("emotional_copy", emotional_questions, mandatory=True)
        
        if self.config.interactive_mode:
            input("Press Enter to complete the questionnaire...")
    
    def get_results(self) -> Dict[str, Any]:
        """Get results as a dictionary (for pipeline use)"""
        # Add metadata
        results = {
            "responses": self.responses,
            "metadata": {
                "timestamp": self.timestamp,
                "total_questions": self.count_total_questions(),
                "completion_status": "completed",
                "fill_mode": self.fill_source,
                "auto_fill_enabled": self.auto_fill_mode,
                "business_type": self.business_type
            }
        }
        
        # Add analysis data if available
        if self.fill_source == "website" and self.website_data:
            results["source_analysis"] = {
                "type": "website",
                "pages_analyzed": len(self.website_data.get('page_summaries', [])),
                "analysis_stats": self.website_data.get('stats', {}),
                "summary_available": bool(self.website_data.get('final_summary'))
            }
        elif self.fill_source == "csv" and self.csv_data:
            results["source_analysis"] = {
                "type": "csv",
                "file_name": self.csv_data.get('file_name', ''),
                "rows_processed": self.csv_data.get('rows_processed', 0),
                "columns_used": self.csv_data.get('columns_used', []),
                "detected_domain": self.csv_data.get('detected_domain', '')
            }
        
        return results
    
    def save_responses(self):
        """Save all responses to a JSON file"""
        if not self.config.save_to_file:
            return None
        
        if self.config.interactive_mode:
            self.clear_screen()
            print("💾 SAVING YOUR RESPONSES")
            print("-" * 30)
        
        results = self.get_results()
        
        # Create filename
        business_name_clean = self.business_type.replace("/", "_").replace(" ", "_").lower()
        mode_suffix = f"_{self.fill_source}fill" if self.auto_fill_mode else "_manual"
        filename = f"sales_questionnaire_{business_name_clean}_{self.timestamp}{mode_suffix}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            self._log(f"✅ Responses saved successfully!")
            self._log(f"📁 File: {filename}")
            self._log(f"📊 Total questions answered: {self.count_answered_questions()}")
            
            if self.auto_fill_mode:
                auto_filled_count = self.count_auto_filled_questions()
                self._log(f"🤖 Auto-filled questions ({self.fill_source}): {auto_filled_count}")
                self._log(f"✏️ Manually edited: {self.count_answered_questions() - auto_filled_count}")
            
            return filename
            
        except Exception as e:
            self._log(f"❌ Error saving file: {e}")
            if self.config.interactive_mode:
                print("\n📋 Here's your data in JSON format:")
                print(json.dumps(results, indent=2, ensure_ascii=False))
            return None
    
    def count_total_questions(self):
        """Count total number of questions asked"""
        count = 5  # Core questions
        count += len(self.get_business_specific_questions())
        count += 2  # Objection handler (if answered)
        count += 3  # Emotional/copy questions
        return count
    
    def count_answered_questions(self):
        """Count how many questions were actually answered"""
        count = 0
        sections = ["core_questions", "business_specific_questions", "objection_handler", "emotional_copy"]
        
        for section in sections:
            for q_data in self.responses.get(section, {}).values():
                if q_data.get("answer", "").strip():
                    count += 1
        
        return count
    
    def count_auto_filled_questions(self):
        """Count how many questions were auto-filled"""
        count = 0
        sections = ["core_questions", "business_specific_questions", "objection_handler", "emotional_copy"]
        
        for section in sections:
            for q_data in self.responses.get(section, {}).values():
                if q_data.get("auto_filled", False) and q_data.get("answer", "").strip():
                    count += 1
        
        return count
    
    def display_summary(self):
        """Display a summary of responses"""
        if not self.config.interactive_mode:
            return
        
        self.clear_screen()
        print("📋 QUESTIONNAIRE SUMMARY")
        print("=" * 50)
        print(f"Business Type: {self.business_type}")
        print(f"Fill Method: {self.get_fill_method_display()}")
        print(f"Questions Answered: {self.count_answered_questions()}")
        if self.auto_fill_mode:
            print(f"Auto-filled: {self.count_auto_filled_questions()}")
        print(f"Completion Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Display source analysis info
        if self.fill_source == "website" and self.website_data:
            print("🌐 WEBSITE ANALYSIS:")
            print(f"• Pages analyzed: {len(self.website_data.get('page_summaries', []))}")
            print(f"• Total processing time: {self.website_data.get('stats', {}).get('time', 0):.1f} seconds")
            print()
        elif self.fill_source == "csv" and self.csv_data:
            print("📊 CSV ANALYSIS:")
            print(f"• File: {self.csv_data.get('file_name', '')}")
            print(f"• Rows processed: {self.csv_data.get('rows_processed', 0)}")
            print(f"• Columns used: {len(self.csv_data.get('columns_used', []))}")
            print(f"• Detected domain: {self.csv_data.get('detected_domain', 'Unknown')}")
            print()
        
        print("🎯 CORE ANSWERS:")
        core_questions = self.responses.get("core_questions", {})
        for q_data in core_questions.values():
            answer_preview = q_data.get("answer", "")[:100]
            if len(q_data.get("answer", "")) > 100:
                answer_preview += "..."
            source_icon = self.get_source_icon(q_data.get("source", "manual"))
            print(f"• {source_icon} {answer_preview}")
        
        print("\n🔥 KEY EMOTIONAL INSIGHTS:")
        emotional_copy = self.responses.get("emotional_copy", {})
        for q_data in emotional_copy.values():
            answer_preview = q_data.get("answer", "")[:100]
            if len(q_data.get("answer", "")) > 100:
                answer_preview += "..."
            source_icon = self.get_source_icon(q_data.get("source", "manual"))
            print(f"• {source_icon} {answer_preview}")
        
        print("\n" + "=" * 50)
        print("🚀 Your sales page questionnaire is complete!")
        print("Use these insights to create compelling copy that converts!")
    
    def get_fill_method_display(self):
        """Get display string for fill method"""
        if self.fill_source == "website":
            return "🌐 Website Auto-fill"
        elif self.fill_source == "csv":
            return "📊 CSV Auto-fill"
        else:
            return "✏️ Manual Entry"
    
    def get_source_icon(self, source):
        """Get icon for answer source"""
        if source == "website":
            return "🌐"
        elif source == "csv":
            return "📊"
        else:
            return "✏️"
    
    def run_pipeline(self) -> Dict[str, Any]:
        """Run questionnaire in pipeline mode (non-interactive)"""
        try:
            # Step 1: Get business type
            self.get_business_type()
            
            # Step 2: Setup auto-fill if configured
            if not self.ask_auto_fill_option():
                raise Exception("Auto-fill setup failed")
            
            # Step 3: Process all questions
            self.ask_core_questions()
            self.ask_business_specific_questions()
            self.ask_objection_handler_questions()
            self.ask_emotional_copy_questions()
            
            # Step 4: Get results
            results = self.get_results()
            
            # Step 5: Save if configured
            filename = self.save_responses()
            if filename:
                results["saved_file"] = filename
            
            return results
            
        except Exception as e:
            self._log(f"❌ Pipeline execution failed: {e}")
            raise
    
    def run(self):
        """Main program execution (interactive mode)"""
        if not self.config.interactive_mode:
            return self.run_pipeline()
        
        self.clear_screen()
        self.display_banner()
        
        try:
            # Step 1: Get business type
            self.get_business_type()
            
            # Step 2: Ask about auto-fill options
            self.ask_auto_fill_option()
            
            # Step 3: Ask core questions
            self.ask_core_questions()
            
            # Step 4: Ask business-specific questions
            self.ask_business_specific_questions()
            
            # Step 5: Ask objection handler questions (optional)
            self.ask_objection_handler_questions()
            
            # Step 6: Ask emotional/copy questions
            self.ask_emotional_copy_questions()
            
            # Step 7: Save responses
            self.save_responses()
            
            # Step 8: Display summary
            self.display_summary()
            
            return self.get_results()
            
        except KeyboardInterrupt:
            print("\n\n⚠️ Questionnaire interrupted by user.")
            save_partial = input("Would you like to save your partial responses? (y/n): ").strip().lower()
            if save_partial in ['y', 'yes']:
                self.responses["metadata"] = {
                    "timestamp": self.timestamp,
                    "completion_status": "partial",
                    "fill_mode": self.fill_source
                }
                self.save_responses()
        
        except Exception as e:
            print(f"\n❌ An unexpected error occurred: {e}")
            print("Your responses will be displayed below:")
            print(json.dumps(self.get_results(), indent=2, ensure_ascii=False))


# Pipeline-friendly functions for main orchestrator
def run_questionnaire_pipeline(config: QuestionnaireConfig) -> Dict[str, Any]:
    """Run questionnaire in pipeline mode - main function for orchestrator"""
    questionnaire = EnhancedSalesPageQuestionnaire(config)
    return questionnaire.run_pipeline()

def create_config(**kwargs) -> QuestionnaireConfig:
    """Create configuration for pipeline use"""
    return QuestionnaireConfig(**kwargs)


if __name__ == "__main__":
    # Interactive mode when run directly
    questionnaire = EnhancedSalesPageQuestionnaire()
    questionnaire.run()

    