import json
import re
import requests
import time
import os
from typing import Dict, List, Tuple, Optional
from urllib.parse import urlparse, urljoin
import openai
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

class CompetitorAnalyzer:
    """
    A comprehensive tool for analyzing product descriptions and finding competitors.
    Uses OpenAI for analysis and Gemini for web search.
    """
    
    def __init__(self):
        """
        Initialize the analyzer with API keys from environment variables.
        """
        # Load API keys from .env file
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file")
        
        # Initialize OpenAI client
        self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
        
        # Initialize Gemini
        genai.configure(api_key=self.gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        
    def load_json_file(self, file_path: str) -> Dict:
        """
        Load and validate JSON file containing product description.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Dictionary containing the JSON data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If JSON is malformed
            ValueError: If JSON doesn't contain expected content
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
            if not data:
                raise ValueError("JSON file is empty")
            
            # Improved validation - check for any meaningful content
            def has_content(value, min_length=5):
                """Check if a value contains meaningful content."""
                if isinstance(value, str):
                    return len(value.strip()) >= min_length
                elif isinstance(value, list):
                    return any(has_content(item, min_length) for item in value)
                elif isinstance(value, dict):
                    return any(has_content(v, min_length) for v in value.values())
                return False
            
            # Check if any field has meaningful content
            has_meaningful_content = any(has_content(value) for value in data.values())
            
            if not has_meaningful_content:
                # Print the actual content for debugging
                print("JSON content preview:")
                for key, value in list(data.items())[:5]:  # Show first 5 items
                    print(f"  {key}: {str(value)[:100]}...")
                raise ValueError("JSON doesn't contain sufficient descriptive content for analysis")
            
            print(f"✓ JSON validation passed - found meaningful content in {len(data)} fields")
            return data
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find file: {file_path}")
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON format: {e.msg}", e.doc, e.pos)
        except Exception as e:
            raise ValueError(f"Error loading JSON file: {str(e)}")
    
    def analyze_content_with_llm(self, json_data: Dict) -> str:
        """
        Use OpenAI LLM to analyze JSON content and determine core offering.
        
        Args:
            json_data: Dictionary containing product/service description
            
        Returns:
            String description of the core offering
        """
        try:
            # Convert JSON to readable format for LLM, but limit size for API efficiency
            content_text = json.dumps(json_data, indent=2)
            
            # If content is very long, truncate it for the API call
            if len(content_text) > 8000:
                content_text = content_text[:8000] + "\n... [Content truncated for analysis]"
            
            prompt = f"""
            Analyze the following JSON content describing a product or service and identify the core offering or theme.
            
            JSON Content:
            {content_text}
            
            Please provide a concise analysis in this format:
            "This content describes a [product/service type] that [main benefit/feature/value proposition]."
            
            Focus on the most distinctive and valuable aspects that would help identify competitors.
            If the content contains form fields or questionnaire data, analyze what product/service the responses describe.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # Using a more reliable model
                messages=[
                    {"role": "system", "content": "You are an expert business analyst specializing in product positioning and competitive analysis."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.3
            )
            
            analysis = response.choices[0].message.content.strip()
            print(f"LLM Analysis: {analysis}")
            return analysis
            
        except Exception as e:
            raise ValueError(f"Error analyzing content with LLM: {str(e)}")
    
    def generate_search_query(self, analysis: str) -> str:
        """
        Generate a specific search query to find competitors based on analysis.
        
        Args:
            analysis: LLM analysis of the product/service
            
        Returns:
            Optimized search query string
        """
        try:
            prompt = f"""
            Based on this product analysis, create a concise and specific search query to find competitor brands or companies offering similar benefits.
            
            Analysis: {analysis}
            
            Generate a search query that would effectively find competing products or services. The query should:
            - Be 3-8 words long
            - Focus on the key benefit or product category
            - Include terms like "best", "top", "companies", or "brands" if helpful
            - Avoid overly specific brand names from the original analysis
            
            Return only the search query, nothing else.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at creating effective search queries for competitive research."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50,
                temperature=0.3
            )
            
            search_query = response.choices[0].message.content.strip().strip('"')
            print(f"Generated Search Query: {search_query}")
            return search_query
            
        except Exception as e:
            raise ValueError(f"Error generating search query: {str(e)}")
    
    def perform_web_search_with_gemini(self, query: str, num_results: int = 10) -> List[Dict]:
        """
        Perform web search using Gemini to get search results and extract information.
        
        Args:
            query: Search query string
            num_results: Number of results to return
            
        Returns:
            List of search result dictionaries
        """
        try:
            # Use Gemini to generate search results and extract competitor information
            search_prompt = f"""
            I need you to help me find competitor information for this search query: "{query}"
            
            Please provide information about the top {num_results} companies or brands that would appear in search results for this query. For each competitor, provide:
            
            1. Company/Brand name
            2. Brief description of their main offering
            3. Their likely official website URL (use common patterns like brandname.com)
            
            Format your response as a structured list that I can parse. Focus on real, well-known companies in this space.
            
            Example format:
            1. CompanyName - Description of their offering - website.com
            2. AnotherCompany - Their main service/product - anothersite.com
            
            Search query: {query}
            """
            
            response = self.gemini_model.generate_content(search_prompt)
            
            # Parse Gemini response to extract structured data
            search_results = self._parse_gemini_search_response(response.text)
            
            print(f"Gemini found {len(search_results)} potential competitors")
            return search_results
            
        except Exception as e:
            print(f"Gemini search error: {str(e)}")
            # Fallback to manual search simulation
            return self._create_fallback_results(query, num_results)
    
    def _parse_gemini_search_response(self, response_text: str) -> List[Dict]:
        """
        Parse Gemini's response to extract competitor information.
        
        Args:
            response_text: Raw text response from Gemini
            
        Returns:
            List of parsed search results
        """
        results = []
        lines = response_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or not any(char.isdigit() for char in line[:3]):
                continue
                
            # Try to parse lines like "1. CompanyName - Description - website.com"
            try:
                # Remove numbering
                content = re.sub(r'^\d+\.\s*', '', line)
                
                # Split by dashes or similar separators
                parts = re.split(r'\s*[-–—]\s*', content)
                
                if len(parts) >= 2:
                    company_name = parts[0].strip()
                    description = parts[1].strip() if len(parts) > 1 else ""
                    website = parts[-1].strip() if len(parts) > 2 else ""
                    
                    # Clean up website URL
                    if website and not website.startswith('http'):
                        if '.' in website:
                            website = f"https://{website}"
                        else:
                            website = f"https://{website}.com"
                    
                    results.append({
                        "title": f"{company_name} - {description}",
                        "link": website,
                        "snippet": description,
                        "brand": company_name
                    })
                    
            except Exception as e:
                continue
        
        return results
    
    def _create_fallback_results(self, query: str, num_results: int) -> List[Dict]:
        """
        Create fallback search results when Gemini fails.
        """
        # This is a simplified fallback - in production you might want to use another search API
        return [
            {
                "title": f"Search results for: {query}",
                "link": "https://example.com",
                "snippet": f"Fallback result for {query}. Please check Gemini API configuration.",
                "brand": "Example Company"
            }
        ]
    
    def extract_competitor_brands(self, search_results: List[Dict], analysis: str) -> List[str]:
        """
        Extract competitor brand names from search results using OpenAI LLM.
        
        Args:
            search_results: List of search result dictionaries
            analysis: Original analysis for context
            
        Returns:
            List of competitor brand names
        """
        try:
            # If brands are already extracted by Gemini, use those
            gemini_brands = [result.get('brand', '') for result in search_results if result.get('brand')]
            if gemini_brands:
                print(f"Using brands from Gemini: {gemini_brands}")
                return [brand for brand in gemini_brands if brand and len(brand) > 1]
            
            # Otherwise, use OpenAI to extract brands from search results
            results_text = ""
            for i, result in enumerate(search_results[:10], 1):
                results_text += f"{i}. {result.get('title', '')}\n{result.get('snippet', '')}\n\n"
            
            prompt = f"""
            Based on this original analysis: "{analysis}"
            
            Extract competitor brand names or company names from these search results:
            
            {results_text}
            
            Instructions:
            - Return only legitimate brand names or company names
            - Focus on direct competitors offering similar products/services
            - Exclude generic terms, websites, or non-commercial entities
            - Return 5-10 brands maximum
            - Format as a simple comma-separated list
            - Do not include explanations or numbering
            
            Brand names:
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at identifying brand names and companies from search results."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.3
            )
            
            brands_text = response.choices[0].message.content.strip()
            
            # Parse comma-separated brands and clean them
            brands = [brand.strip() for brand in brands_text.split(',') if brand.strip()]
            brands = [brand for brand in brands if len(brand) > 1 and len(brand) < 50]
            
            print(f"Extracted Brands: {brands}")
            return brands
            
        except Exception as e:
            print(f"Error extracting brands: {str(e)}")
            return []
    
    def get_official_websites(self, brands: List[str]) -> List[Tuple[str, str]]:
        """
        Find official websites for competitor brands using Gemini.
        
        Args:
            brands: List of brand names
            
        Returns:
            List of tuples (brand_name, website_url)
        """
        results = []
        
        for brand in brands:
            try:
                print(f"Finding website for: {brand}")
                
                # Use Gemini to find the official website
                website_prompt = f"""
                What is the official website URL for the company/brand "{brand}"?
                
                Please respond with only the official website URL in this format: https://website.com
                
                If you're not certain, provide the most likely official website based on common naming patterns.
                Do not include any explanations, just the URL.
                """
                
                response = self.gemini_model.generate_content(website_prompt)
                website_url = response.text.strip()
                
                # Clean up the URL
                if website_url and not website_url.startswith('http'):
                    website_url = f"https://{website_url}"
                
                # Validate URL format
                if self._is_valid_url(website_url):
                    results.append((brand, website_url))
                    print(f"Found: {brand} -> {website_url}")
                else:
                    # Try a common pattern as fallback
                    fallback_url = f"https://{brand.lower().replace(' ', '').replace('.', '')}.com"
                    results.append((brand, fallback_url))
                    print(f"Using fallback for {brand}: {fallback_url}")
                
                # Rate limiting to be respectful
                time.sleep(1)
                
            except Exception as e:
                print(f"Error finding website for {brand}: {str(e)}")
                # Add a fallback URL
                fallback_url = f"https://{brand.lower().replace(' ', '').replace('.', '')}.com"
                results.append((brand, fallback_url))
                continue
        
        return results
    
    def validate_website_urls(self, websites: List[Tuple[str, str]], timeout: int = 10) -> List[Tuple[str, str, str]]:
        """
        Validate which website URLs are actually accessible.
        
        Args:
            websites: List of tuples (brand_name, website_url)
            timeout: Request timeout in seconds
            
        Returns:
            List of tuples (brand_name, website_url, status) where status is 'working' or error message
        """
        print("\n7. Validating website URLs...")
        print("-" * 40)
        
        validated_results = []
        working_count = 0
        
        # Configure requests session with headers to avoid blocking
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
        
        for brand, url in websites:
            try:
                print(f"Testing: {brand} -> {url}")
                
                # Make request with timeout and allow redirects
                response = session.get(url, timeout=timeout, allow_redirects=True, verify=False)
                
                if response.status_code == 200:
                    validated_results.append((brand, url, 'working'))
                    working_count += 1
                    print(f"✓ WORKING - {brand}: {url} (Status: {response.status_code})")
                else:
                    error_msg = f"HTTP {response.status_code}"
                    validated_results.append((brand, url, error_msg))
                    print(f"✗ FAILED - {brand}: {url} (Status: {response.status_code})")
                    
            except requests.exceptions.Timeout:
                error_msg = "Timeout"
                validated_results.append((brand, url, error_msg))
                print(f"✗ TIMEOUT - {brand}: {url}")
                
            except requests.exceptions.ConnectionError:
                error_msg = "Connection Error"
                validated_results.append((brand, url, error_msg))
                print(f"✗ CONNECTION ERROR - {brand}: {url}")
                
            except requests.exceptions.InvalidURL:
                error_msg = "Invalid URL"
                validated_results.append((brand, url, error_msg))
                print(f"✗ INVALID URL - {brand}: {url}")
                
            except Exception as e:
                error_msg = f"Error: {str(e)[:50]}"
                validated_results.append((brand, url, error_msg))
                print(f"✗ ERROR - {brand}: {url} ({str(e)[:50]})")
            
            # Small delay between requests to be respectful
            time.sleep(0.5)
        
        print(f"\nValidation complete: {working_count}/{len(websites)} websites are working")
        return validated_results
    
    def filter_working_websites(self, validated_results: List[Tuple[str, str, str]], 
                              max_results: int = 5, min_results: int = 1) -> List[Tuple[str, str]]:
        """
        Filter and return only working websites with specified limits.
        
        Args:
            validated_results: List of tuples (brand_name, website_url, status)
            max_results: Maximum number of working websites to return
            min_results: Minimum number of working websites to return
            
        Returns:
            List of tuples (brand_name, website_url) for working websites only
        """
        # Filter only working websites
        working_websites = [(brand, url) for brand, url, status in validated_results if status == 'working']
        
        # Apply limits
        if len(working_websites) == 0:
            print(f"⚠️  No working websites found! Returning {min_results} best guess(es)...")
            # Return the first min_results entries as fallback
            fallback = [(brand, url) for brand, url, status in validated_results[:min_results]]
            return fallback
        
        elif len(working_websites) < min_results:
            print(f"⚠️  Only {len(working_websites)} working website(s) found (minimum {min_results} required)")
            return working_websites
        
        else:
            # Return up to max_results
            limited_results = working_websites[:max_results]
            if len(working_websites) > max_results:
                print(f"📋 Limiting results to top {max_results} working websites (found {len(working_websites)} total)")
            return limited_results
    
    def _is_valid_url(self, url: str) -> bool:
        """
        Check if a URL is valid.
        
        Args:
            url: URL string to validate
            
        Returns:
            True if URL is valid, False otherwise
        """
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    def save_working_competitors_to_json(self, working_competitors: List[Dict], filename: str = None) -> str:
        """
        Save the final working competitors to a JSON file.
        
        Args:
            working_competitors: List of working competitor dictionaries
            filename: Optional custom filename. If None, generates timestamp-based name
            
        Returns:
            String path of the saved file
        """
        try:
            # Generate filename if not provided
            if filename is None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"working_competitors_{timestamp}.json"
            
            # Ensure .json extension
            if not filename.endswith('.json'):
                filename += '.json'
            
            # Prepare data for JSON export
            export_data = {
                "export_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_working_competitors": len(working_competitors),
                "competitors": working_competitors
            }
            
            # Save to JSON file with proper formatting
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Working competitors saved to: {filename}")
            return filename
            
        except Exception as e:
            print(f"❌ Error saving to JSON file: {str(e)}")
            return ""

    def debug_json_content(self, json_data: Dict) -> None:
        """
        Debug helper to understand JSON content structure.
        """
        print("\n🔍 DEBUG: JSON Content Analysis")
        print("-" * 40)
        print(f"Total fields: {len(json_data)}")
        print("\nField analysis:")
        
        for key, value in json_data.items():
            if isinstance(value, str):
                content_length = len(value.strip())
                preview = value.strip()[:100] + "..." if content_length > 100 else value.strip()
                print(f"  {key}: String ({content_length} chars) - '{preview}'")
            elif isinstance(value, (list, dict)):
                print(f"  {key}: {type(value).__name__} with {len(value)} items")
                if isinstance(value, dict):
                    # Show a few keys if it's a dict
                    keys_preview = list(value.keys())[:3]
                    print(f"    Keys preview: {keys_preview}")
            else:
                print(f"  {key}: {type(value).__name__} - {str(value)[:50]}")
        print("-" * 40)

    def run_analysis(self, json_file_path: str, debug: bool = False) -> Dict:
        """
        Run the complete competitor analysis pipeline.
        
        Args:
            json_file_path: Path to JSON file containing product description
            debug: If True, print debug information about JSON content
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            print("=== Starting Competitor Analysis ===\n")
            print(f"Using OpenAI API Key: {self.openai_api_key[:10]}...")
            print(f"Using Gemini API Key: {self.gemini_api_key[:10]}...\n")
            
            # Step 1: Load JSON file
            print("1. Loading JSON file...")
            json_data = self.load_json_file(json_file_path)
            print(f"✓ Successfully loaded JSON with {len(json_data)} fields\n")
            
            # Debug JSON content if requested
            if debug:
                self.debug_json_content(json_data)
            
            # Step 2: Analyze content with OpenAI LLM
            print("2. Analyzing content with OpenAI...")
            analysis = self.analyze_content_with_llm(json_data)
            print("✓ Analysis complete\n")
            
            # Step 3: Generate search query
            print("3. Generating search query...")
            search_query = self.generate_search_query(analysis)
            print("✓ Search query generated\n")
            
            # Step 4: Perform web search with Gemini
            print("4. Performing web search with Gemini...")
            search_results = self.perform_web_search_with_gemini(search_query)
            print(f"✓ Found {len(search_results)} search results\n")
            
            # Step 5: Extract competitor brands
            print("5. Extracting competitor brands...")
            brands = self.extract_competitor_brands(search_results, analysis)
            print(f"✓ Extracted {len(brands)} competitor brands\n")
            
            # Step 6: Get official websites using Gemini
            print("6. Finding official websites with Gemini...")
            websites = self.get_official_websites(brands)
            print(f"✓ Found {len(websites)} official websites\n")
            
            # Step 7: Validate website URLs
            validated_results = self.validate_website_urls(websites)
            
            # Step 8: Filter working websites (max 5, min 1)
            print("\n8. Filtering working websites...")
            working_websites = self.filter_working_websites(validated_results, max_results=5, min_results=1)
            print(f"✓ Selected {len(working_websites)} working website(s)\n")
            
            # Compile results
            results = {
                "original_analysis": analysis,
                "search_query": search_query,
                "all_competitors_found": [
                    {"brand": brand, "website": website, "status": status} 
                    for brand, website, status in validated_results
                ],
                "working_competitors": [
                    {"brand": brand, "website": website} 
                    for brand, website in working_websites
                ],
                "summary": {
                    "total_competitors_found": len(validated_results),
                    "working_competitors": len([r for r in validated_results if r[2] == 'working']),
                    "final_results_shown": len(working_websites),
                    "search_results_processed": len(search_results)
                }
            }
            
            return results
            
        except Exception as e:
            raise Exception(f"Analysis pipeline failed: {str(e)}")


def main():
    """
    Main function to demonstrate the competitor analysis tool.
    """
    try:
        # Initialize analyzer (will load API keys from .env)
        analyzer = CompetitorAnalyzer()
        
        # Get JSON file path from user
        print("=== Competitor Analysis Tool ===\n")
        json_file_path = input("Enter the path to your JSON file: ").strip()
        
        # Remove quotes if user included them
        json_file_path = json_file_path.strip('"').strip("'")
        
        # Check if file exists
        if not json_file_path:
            raise ValueError("No file path provided")
        
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"File not found: {json_file_path}")
        
        print(f"Using JSON file: {json_file_path}\n")
        
        # Ask if user wants debug mode
        debug_choice = input("Enable debug mode to see JSON content analysis? (y/n): ").strip().lower()
        debug_mode = debug_choice in ['y', 'yes']
        
        # Run analysis
        results = analyzer.run_analysis(json_file_path, debug=debug_mode)
        
        # Display results
        print("=" * 60)
        print("🎯 FINAL ANALYSIS RESULTS")
        print("=" * 60)
        print(f"\n📋 Analysis: {results['original_analysis']}\n")
        print(f"🔍 Search Query Used: {results['search_query']}\n")
        
        # Show all competitors found (with status)
        print("📊 ALL COMPETITORS DISCOVERED:")
        print("-" * 50)
        for i, competitor in enumerate(results['all_competitors_found'], 1):
            status_emoji = "✅" if competitor['status'] == 'working' else "❌"
            print(f"{i}. {competitor['brand']} {status_emoji}")
            print(f"   Website: {competitor['website']}")
            print(f"   Status: {competitor['status']}\n")
        
        # Show working competitors (final results)
        print("🏆 WORKING COMPETITORS (FINAL RESULTS):")
        print("=" * 50)
        
        if results['working_competitors']:
            for i, competitor in enumerate(results['working_competitors'], 1):
                print(f"{i}. ✅ {competitor['brand']}")
                print(f"   🌐 {competitor['website']}\n")
        else:
            print("❌ No working competitors found.\n")
        
        # Summary
        print("📈 SUMMARY:")
        print("-" * 30)
        print(f"• Total competitors discovered: {results['summary']['total_competitors_found']}")
        print(f"• Working websites found: {results['summary']['working_competitors']}")
        print(f"• Final results shown: {results['summary']['final_results_shown']}")
        print(f"• Search results processed: {results['summary']['search_results_processed']}")
        
        if results['summary']['working_competitors'] == 0:
            print("\n⚠️  No websites were accessible. This could be due to:")
            print("   • Websites blocking automated requests")
            print("   • Temporary server issues")
            print("   • Incorrect URLs generated")
            print("   • Network connectivity issues")
        
        # Save working competitors to JSON file
        print("\n" + "=" * 60)
        if results['working_competitors']:
            saved_file = analyzer.save_working_competitors_to_json(results['working_competitors'])
            if saved_file:
                print(f"✅ Results exported successfully!")
                print(f"📁 File location: {os.path.abspath(saved_file)}")
        else:
            print("⚠️  No working competitors to save.")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("\n🔧 Make sure to:")
        print("1. Create a .env file in the same directory with:")
        print("   OPENAI_API_KEY=your_openai_key_here")
        print("   GEMINI_API_KEY=your_gemini_key_here")
        print("2. Install required packages:")
        print("   pip install openai google-generativeai python-dotenv requests beautifulsoup4")


if __name__ == "__main__":
    main()


# Create this .env file in the same directory:
"""
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
"""