
#!/usr/bin/env python3
"""
Smart Web Scraper with Intelligent Link Validation & Circuit Breaker

===================================================================

A much smarter web scraper that:
- Validates URLs before attempting to scrape them
- Respects robots.txt
- Uses circuit breaker pattern for failing endpoints
- Implements smart rate limiting based on server responses
- Filters out non-content URLs intelligently
- Uses session management for better performance
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, parse_qs, quote
from urllib.robotparser import RobotFileParser
import time
import random
import os
import re
from datetime import datetime, timedelta
from openai import OpenAI
from dotenv import load_dotenv
import concurrent.futures
import threading
from collections import deque, defaultdict
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import hashlib
import json

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load environment variables
load_dotenv()

class CircuitBreaker:
    """Circuit breaker to prevent hammering failing endpoints"""
    
    def __init__(self, failure_threshold=5, recovery_timeout=300):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half-open
    
    def call(self, func, *args, **kwargs):
        if self.state == 'open':
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = 'half-open'
            else:
                raise Exception("Circuit breaker is OPEN - too many failures")
        
        try:
            result = func(*args, **kwargs)
            if self.state == 'half-open':
                self.state = 'closed'
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = 'open'
                print(f"🚨 Circuit breaker OPENED - too many failures")
            
            raise e

class SmartURLValidator:
    """Smart URL validation and filtering"""
    
    def __init__(self, base_domain):
        self.base_domain = base_domain.replace('www.', '').lower()
        
        # URLs that are definitely not content
        self.skip_patterns = [
            # Cloudflare and CDN
            r'/cdn-cgi/',
            r'/cf-ray/',
            
            # Technical files
            r'\.(css|js|ico|png|jpg|jpeg|gif|svg|woff|woff2|ttf|eot)$',
            r'\.(pdf|doc|docx|xls|xlsx|zip|rar|mp4|mp3|avi|mov)$',
            r'\.json$',
            r'\.xml$',
            r'\.txt$',
            
            # Admin and system paths
            r'/wp-admin/',
            r'/wp-content/uploads/',
            r'/admin/',
            r'/_next/static/',
            r'/static/',
            r'/assets/',
            r'/node_modules/',
            r'/.well-known/',
            
            # Social media and external
            r'facebook\.com',
            r'twitter\.com',
            r'linkedin\.com',
            r'instagram\.com',
            r'youtube\.com',
            r'mailto:',
            r'tel:',
            r'javascript:',
            r'#',
            
            # Tracking and analytics
            r'google-analytics',
            r'googletagmanager',
            r'facebook\.net',
            r'doubleclick\.net',
            
            # Common non-content patterns
            r'/search\?',
            r'\?utm_',
            r'/feed/?$',
            r'/rss/?$',
            r'/sitemap',
        ]
        
        # Patterns that usually contain good content
        self.good_patterns = [
            r'/blog/',
            r'/article/',
            r'/post/',
            r'/news/',
            r'/about/',
            r'/contact/',
            r'/service/',
            r'/product/',
            r'/pricing/',
            r'/feature/',
            r'/help/',
            r'/support/',
            r'/docs/',
            r'/guide/',
            r'/tutorial/',
        ]
        
        self.compiled_skip = [re.compile(pattern, re.IGNORECASE) for pattern in self.skip_patterns]
        self.compiled_good = [re.compile(pattern, re.IGNORECASE) for pattern in self.good_patterns]
    
    def is_valid_url(self, url):
        """Comprehensive URL validation"""
        if not url or len(url) > 2000:
            return False
            
        try:
            parsed = urlparse(url)
        except:
            return False
        
        # Must be HTTP/HTTPS
        if parsed.scheme not in ['http', 'https']:
            return False
        
        # Must be same domain
        domain = parsed.netloc.replace('www.', '').lower()
        if domain != self.base_domain:
            return False
        
        # Check skip patterns
        full_url = url.lower()
        for pattern in self.compiled_skip:
            if pattern.search(full_url):
                return False
        
        # Avoid URLs with too many parameters (likely tracking/filters)
        if len(parsed.query) > 200:
            return False
        
        # Avoid very deep paths (likely not main content)
        path_depth = len([p for p in parsed.path.split('/') if p])
        if path_depth > 6:
            return False
        
        return True
    
    def get_url_priority(self, url):
        """Assign priority to URLs (higher = better content)"""
        priority = 0
        url_lower = url.lower()
        
        # Boost for good content patterns
        for pattern in self.compiled_good:
            if pattern.search(url_lower):
                priority += 10
        
        # Boost for shorter paths (usually main pages)
        path_depth = len([p for p in urlparse(url).path.split('/') if p])
        priority += max(0, 5 - path_depth)
        
        # Boost for no parameters
        if not urlparse(url).query:
            priority += 2
        
        return priority

class SmartWebScraper:
    """Intelligent web scraper with proper error handling and validation"""
    
    def __init__(self, base_url, openai_api_key=None, max_workers=2, max_pages=50):
        self.base_url = base_url.rstrip('/')
        parsed = urlparse(base_url)
        self.domain = parsed.netloc
        self.base_domain = parsed.netloc.replace('www.', '')
        
        self.max_workers = max_workers
        self.max_pages = max_pages
        
        # Thread-safe collections
        self.visited = set()
        self.failed_urls = set()
        self.page_summaries = []
        self.lock = threading.Lock()
        
        # Smart URL management
        self.url_validator = SmartURLValidator(self.base_domain)
        self.priority_urls = deque()  # High priority URLs
        self.regular_urls = deque()   # Regular URLs
        
        # Circuit breakers for different types of failures
        self.circuit_breakers = {
            'timeout': CircuitBreaker(failure_threshold=3, recovery_timeout=180),
            'server_error': CircuitBreaker(failure_threshold=5, recovery_timeout=300),
            'rate_limit': CircuitBreaker(failure_threshold=2, recovery_timeout=600)
        }
        
        # Rate limiting
        self.last_request_time = defaultdict(float)
        self.min_delay = 2.0  # Minimum delay between requests
        self.adaptive_delay = 2.0  # Adaptive delay that increases with errors
        
        # Session management
        self.session = self.create_smart_session()
        
        # Robots.txt compliance
        self.robots_parser = None
        self.check_robots_txt()
        
        # OpenAI client
        if openai_api_key:
            self.openai_client = OpenAI(api_key=openai_api_key)
        else:
            self.openai_client = None
        
        # Add initial URLs
        self.add_url(base_url, priority=10)
        
        # Statistics
        self.stats = {
            'requests_made': 0,
            'errors_encountered': 0,
            'urls_filtered': 0,
            'rate_limit_hits': 0,
            'circuit_breaker_trips': 0
        }
    
    def create_smart_session(self):
        """Create a session with proper retry strategy"""
        session = requests.Session()
        
        # Retry strategy - be more conservative
        try:
            # Try newer parameter name first
            retry_strategy = Retry(
                total=2,  # Reduced retries
                backoff_factor=2,
                status_forcelist=[429, 500, 502, 503, 504, 522],
                allowed_methods=["HEAD", "GET", "OPTIONS"]
            )
        except TypeError:
            # Fall back to older parameter name
            retry_strategy = Retry(
                total=2,  # Reduced retries
                backoff_factor=2,
                status_forcelist=[429, 500, 502, 503, 504, 522],
                method_whitelist=["HEAD", "GET", "OPTIONS"]
            )
        
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=20)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def check_robots_txt(self):
        """Check and respect robots.txt"""
        try:
            robots_url = f"{self.base_url}/robots.txt"
            self.robots_parser = RobotFileParser()
            self.robots_parser.set_url(robots_url)
            self.robots_parser.read()
            print(f"✓ Loaded robots.txt from {robots_url}")
        except Exception as e:
            print(f"⚠️  Could not load robots.txt: {e}")
            self.robots_parser = None
    
    def can_fetch_url(self, url):
        """Check if we can fetch this URL according to robots.txt"""
        if not self.robots_parser:
            return True
        
        try:
            return self.robots_parser.can_fetch('*', url)
        except:
            return True  # If in doubt, allow it
    
    def add_url(self, url, priority=0):
        """Add URL to appropriate queue based on priority"""
        if not self.url_validator.is_valid_url(url):
            with self.lock:
                self.stats['urls_filtered'] += 1
            return
        
        normalized_url = self.normalize_url(url)
        
        with self.lock:
            if normalized_url in self.visited or normalized_url in self.failed_urls:
                return
            
            # Calculate priority
            actual_priority = priority + self.url_validator.get_url_priority(url)
            
            if actual_priority >= 8:
                if normalized_url not in self.priority_urls:
                    self.priority_urls.append(normalized_url)
            else:
                if normalized_url not in self.regular_urls:
                    self.regular_urls.append(normalized_url)
    
    def get_next_url(self):
        """Get next URL to process, prioritizing high-priority URLs"""
        with self.lock:
            if len(self.page_summaries) >= self.max_pages:
                return None
            
            # Try priority queue first
            if self.priority_urls:
                return self.priority_urls.popleft()
            
            # Then regular queue
            if self.regular_urls:
                return self.regular_urls.popleft()
            
            return None
    
    def normalize_url(self, url):
        """Normalize URL for deduplication"""
        if not url:
            return ""
        
        parsed = urlparse(url)
        
        # Remove fragment
        path = parsed.path.rstrip('/')
        if not path:
            path = '/'
        
        # Remove common tracking parameters
        query_params = parse_qs(parsed.query)
        clean_params = {}
        
        # Keep only important parameters
        important_params = ['id', 'page', 'category', 'tag', 'slug', 'section']
        for key, values in query_params.items():
            if key.lower() in important_params and values:
                clean_params[key] = values[0]
        
        # Rebuild query string
        if clean_params:
            query_pairs = [f"{key}={quote(str(value))}" for key, value in clean_params.items()]
            query = '&'.join(query_pairs)
        else:
            query = ''
        
        return f"{parsed.scheme}://{parsed.netloc}{path}{'?' + query if query else ''}"
    
    def smart_delay(self):
        """Implement smart delays based on server responses"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time[self.domain]
        
        delay_needed = self.adaptive_delay - time_since_last
        if delay_needed > 0:
            time.sleep(delay_needed)
        
        self.last_request_time[self.domain] = time.time()
    
    def scrape_page(self, url):
        """Scrape a single page with comprehensive error handling"""
        normalized_url = self.normalize_url(url)
        
        with self.lock:
            if normalized_url in self.visited:
                return None
            self.visited.add(normalized_url)
        
        # Check robots.txt
        if not self.can_fetch_url(url):
            print(f"🚫 Robots.txt disallows: {url}")
            return None
        
        # Smart delay before request
        self.smart_delay()
        
        try:
            # Use circuit breaker pattern
            result = self.circuit_breakers['server_error'].call(self._make_request, url)
            return result
            
        except Exception as e:
            error_msg = str(e)
            
            with self.lock:
                self.stats['errors_encountered'] += 1
                self.failed_urls.add(normalized_url)
            
            if "Circuit breaker is OPEN" in error_msg:
                with self.lock:
                    self.stats['circuit_breaker_trips'] += 1
                print(f"🚨 Circuit breaker prevented request to {url}")
                # Increase delay when circuit breaker trips
                self.adaptive_delay = min(self.adaptive_delay * 1.5, 10.0)
            elif "429" in error_msg or "rate limit" in error_msg.lower():
                with self.lock:
                    self.stats['rate_limit_hits'] += 1
                print(f"🐌 Rate limited: {url}")
                # Significant delay increase for rate limiting
                self.adaptive_delay = min(self.adaptive_delay * 2, 15.0)
            else:
                print(f"❌ Failed to scrape {url}: {error_msg[:100]}")
            
            return None
    
    def _make_request(self, url):
        """Make the actual HTTP request"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0',
            'DNT': '1'
        }
        
        with self.lock:
            self.stats['requests_made'] += 1
        
        response = self.session.get(url, headers=headers, timeout=30, verify=False)
        
        # Handle specific status codes
        if response.status_code == 429:
            raise Exception("Rate limited (429)")
        elif response.status_code in [522, 503, 504]:
            raise Exception(f"Server error ({response.status_code})")
        
        response.raise_for_status()
        
        # Reset adaptive delay on success
        self.adaptive_delay = max(self.adaptive_delay * 0.9, self.min_delay)
        
        return self._process_response(url, response)
    
    def _process_response(self, url, response):
        """Process the HTTP response"""
        content_type = response.headers.get('content-type', '').lower()
        if 'text/html' not in content_type:
            return None
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract links BEFORE removing elements
        links = self._extract_links(soup, url)
        
        # Clean up HTML for content extraction
        for element in soup(['script', 'style', 'noscript', 'iframe', 'svg', 'header', 'nav', 'footer', 'aside']):
            element.decompose()
        
        # Get title
        title_elem = soup.find('title')
        title = title_elem.get_text().strip() if title_elem else "No Title"
        
        # Extract main content
        content = self._extract_content(soup)
        
        # Add new valid links
        valid_links = []
        for link in links:
            if self.url_validator.is_valid_url(link):
                valid_links.append(link)
                self.add_url(link)
        
        page_count = len(self.visited)
        word_count = len(content.split())
        
        print(f"✓ [{page_count:2d}] {url}")
        print(f"    📊 {word_count} words | {len(valid_links)} valid links found")
        
        return {
            'url': url,
            'title': title,
            'content': content,
            'word_count': word_count,
            'links_found': len(valid_links)
        }
    
    def _extract_links(self, soup, base_url):
        """Extract all valid links from the page"""
        links = []
        
        for element in soup.find_all(['a', 'area'], href=True):
            href = element.get('href', '').strip()
            if not href or href.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
                continue
            
            full_url = urljoin(base_url, href)
            links.append(full_url)
        
        return links
    
    def _extract_content(self, soup):
        """Extract main content from cleaned HTML"""
        # Try to find main content areas
        main_selectors = [
            'main', 'article', '[role="main"]', '.main-content',
            '.content', '.post-content', '.entry-content', '.article-content'
        ]
        
        main_content = None
        for selector in main_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        if not main_content:
            main_content = soup.find('body') or soup
        
        # Convert to text
        for br in main_content.find_all("br"):
            br.replace_with("\n")
        
        text = main_content.get_text(separator='\n', strip=True)
        
        # Clean and filter text
        lines = []
        for line in text.split('\n'):
            line = line.strip()
            if len(line) > 3 and not line.isdigit():  # Skip very short lines and numbers
                lines.append(line)
        
        return '\n'.join(lines)
    
    def summarize_page(self, page_data):
        """Summarize page content using OpenAI"""
        if not self.openai_client:
            return {
                'url': page_data['url'],
                'title': page_data['title'],
                'summary': f"Content preview:\n{page_data['content'][:300]}...",
                'word_count': page_data['word_count']
            }
        
        content = page_data['content'][:4000]  # Limit content for API
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {"role": "system", "content": "Create a concise, informative summary of this webpage content."},
                    {"role": "user", "content": f"Title: {page_data['title']}\nURL: {page_data['url']}\n\nContent:\n{content}"}
                ],
                max_tokens=300,
                temperature=0.1
            )
            
            summary = response.choices[0].message.content.strip()
            
        except Exception as e:
            summary = f"Summary failed: {str(e)[:50]}\n\nContent preview:\n{content[:200]}..."
        
        return {
            'url': page_data['url'],
            'title': page_data['title'],
            'summary': summary,
            'word_count': page_data['word_count']
        }
    
    def process_url(self, url):
        """Process a single URL completely"""
        page_data = self.scrape_page(url)
        if not page_data:
            return None
        
        summary_data = self.summarize_page(page_data)
        
        with self.lock:
            self.page_summaries.append(summary_data)
        
        return summary_data
    
    def create_final_summary(self):
        """Create comprehensive final summary"""
        if not self.openai_client:
            return self.create_basic_summary()
        
        print(f"\n🔄 Creating final summary from {len(self.page_summaries)} pages...")
        
        # Prepare summary content (limit to avoid token limits)
        summaries_text = []
        for i, summary in enumerate(self.page_summaries[:25], 1):
            summaries_text.append(f"Page {i}: {summary['title']}\nURL: {summary['url']}\nSummary: {summary['summary'][:300]}...")
        
        combined_text = "\n\n".join(summaries_text)
        
        try:
            prompt = f"""Create a comprehensive final summary of this website based on the individual page summaries below.

Website: {self.base_domain}
Total Pages Analyzed: {len(self.page_summaries)}

Page Summaries:
{combined_text}

Create a structured final summary with:

1. **Website Overview**: What is this website about and its main purpose?

2. **Key Topics & Content**: What are the main themes and subject areas covered?

3. **Important Information**: Significant facts, data, products, or services mentioned.

4. **Content Organization**: How is the website structured and what types of pages exist?

5. **Notable Entities**: Important companies, people, products, or concepts referenced.

6. **Overall Assessment**: What type of website this is and its primary value proposition."""

            response = self.openai_client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing websites and creating comprehensive summaries from multiple page analyses."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.1
            )
            
            final_summary = response.choices[0].message.content.strip()
            print("✓ Final comprehensive summary created")
            return final_summary
            
        except Exception as e:
            print(f"✗ Final summary failed: {e}")
            return self.create_basic_summary()

    def create_marketing_analysis(self, final_summary):
        """Create marketing-focused analysis in the specified format"""
        if not self.openai_client:
            return self.create_basic_marketing_analysis()
        
        print(f"🎯 Creating marketing analysis...")
        
        # Prepare content for analysis
        all_content = f"""Final Summary:
{final_summary}

Individual Page Summaries:
"""
        
        for summary in self.page_summaries[:15]:  # Use first 15 to avoid token limits
            all_content += f"\n- {summary['title']}: {summary['summary'][:200]}..."
        
        try:
            prompt = f"""
You are a meticulous marketing strategist analyzing a website for sales page creation. Based on the website content below, provide a **detailed, structured breakdown** in the **EXACT format requested**.

🔒 VERY IMPORTANT RULES:
- Use **actual phrases, copy, and claims from the site** where possible.
- If any section lacks clear info, **write: "Not found in analysis"** — do NOT assume or invent.
- Each section should be **detailed, not just one-liners**.
- Your output will be used to generate persuasive sales copy, so depth and clarity are critical.

🖥️ Website being analyzed: {self.base_domain}

📄 Website Content for Analysis:
{all_content}

---

🎯 STRUCTURE YOUR OUTPUT IN THIS EXACT FORMAT BELOW:
(Answer each section thoroughly and separately)

🔹 1. **Elevator Pitch / One-liner**
- Write a concise sentence explaining what the business does and for whom.
- Follow this format: “We help [target customer] do [job/result] using [product/service].”
- Provide multiple versions if the message is unclear or fragmented.

🔹 2. **Core Product/Service Description**
- Type of offering (e.g., SaaS, course, service, platform, physical product)?
- Who is the target user?
- List all major features (bullet points).
- Mention any pricing, tiers, bundles, or offers if visible.
- Note any guarantees, trials, or bonuses mentioned.

🔹 3. **The Problem It Solves**
- What key pain points or frustrations are mentioned or implied before the user discovers the product?
- Quote exact problem statements or phrases from the content when possible.

🔹 4. **The Transformation / Outcome**
- What practical and emotional transformation does the user experience after using the product/service?
- Use direct phrases from testimonials or benefit-focused copy.

🔹 5. **Proof & Credibility**
- List all available credibility elements (testimonials, reviews, case studies, user counts, media mentions, client logos).
- If nothing is found, explicitly state "Not found in analysis".

🔹 6. **Unique Selling Points / Differentiators**
- What makes this offer different from others in the same market?
- Are any features, methods, or results exclusive or rare?

🔹 7. **Target Customer Profile**
- Clearly describe the audience (demographics, roles, business type, psychographics).
- B2B or B2C? Solopreneurs, founders, enterprise teams, etc.?

🔹 8. **Objection Killers**
- List any FAQ content, claims, or guarantees that address potential customer hesitations.
- Note any urgency, limited-time offers, or money-back claims.

🔹 9. **First User Experience (for SaaS/platforms/services)**
- Describe what happens immediately after signup or purchase.
- Is there onboarding? Setup guidance? A key action?

🔹 10. **Bonus Emotional Gold**
- What emotional desire or aspiration does the product tap into?
- Are any beliefs challenged or reversed by using this product?
- What fear, regret, or FOMO might customers feel if they don’t act?

---

📌 Final Notes:
- Format your output cleanly with headers and bullet points.
- Be exhaustive wherever content allows.
- Never guess. Only extract and format what’s clearly present.
"""

            response = self.openai_client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {"role": "system", "content": "You are an expert marketing analyst who creates detailed breakdowns of business websites for competitive analysis and copywriting insights."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.1
            )
            
            marketing_analysis = response.choices[0].message.content.strip()
            print("✓ Marketing analysis created")
            return marketing_analysis
            
        except Exception as e:
            print(f"✗ Marketing analysis failed: {e}")
            return self.create_basic_marketing_analysis()

    def create_basic_marketing_analysis(self):
        """Create basic marketing analysis without AI"""
        return f"""# Marketing Analysis: {self.base_domain}

**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Pages Analyzed:** {len(self.page_summaries)}

🔹 1. Elevator Pitch / One-liner
* Manual analysis required - AI not available

🔹 2. Core Product/Service Description  
* Analysis based on {len(self.page_summaries)} pages from {self.base_domain}
* Manual review of saved summaries recommended

🔹 3. The Problem It Solves
* Detailed analysis requires AI summarization

🔹 4. The Transformation / Outcome
* Review individual page summaries for outcome statements

🔹 5. Proof & Credibility
* Check saved summaries for testimonials and social proof

🔹 6. Unique Selling Points / Differentiators
* Manual analysis of competitive positioning needed

🔹 7. Target Customer Profile
* Review page content for customer language and use cases

🔹 8. Objection Killers
* Look for FAQ pages and "Why choose us" content in summaries

🔹 9. First User Experience
* Check for onboarding or getting started pages

🔹 10. Bonus Emotional Gold
* Manual review of messaging and emotional language recommended

**Note:** For detailed marketing analysis, add OpenAI API key to environment."""

    def create_basic_summary(self):
        """Create basic summary without AI"""
        total_words = sum(s['word_count'] for s in self.page_summaries)
        
        summary = f"""# Website Analysis: {self.base_domain}

**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Pages Analyzed:** {len(self.page_summaries)}
**Total Word Count:** {total_words:,}

## Website Overview
This analysis covers {len(self.page_summaries)} pages from {self.base_domain}, containing approximately {total_words:,} words of content.

## Pages Analyzed:
"""
        
        for i, summary in enumerate(self.page_summaries, 1):
            summary_text = summary['summary'][:200]
            if len(summary['summary']) > 200:
                summary_text += '...'
            
            summary += f"""
### {i}. {summary['title']}
**URL:** {summary['url']}
**Word Count:** {summary['word_count']:,}
**Summary:** {summary_text}
"""
        
        return summary
        """Create basic summary without AI"""
        total_words = sum(s['word_count'] for s in self.page_summaries)
        
        summary = f"""# Website Analysis: {self.base_domain}

**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Pages Analyzed:** {len(self.page_summaries)}
**Total Word Count:** {total_words:,}

## Website Overview
This analysis covers {len(self.page_summaries)} pages from {self.base_domain}, containing approximately {total_words:,} words of content.

## Pages Analyzed:
"""
        
        for i, summary in enumerate(self.page_summaries, 1):
            summary_text = summary['summary'][:200]
            if len(summary['summary']) > 200:
                summary_text += '...'
            
            summary += f"""
### {i}. {summary['title']}
**URL:** {summary['url']}
**Word Count:** {summary['word_count']:,}
**Summary:** {summary_text}
"""
        
        return summary

    def save_results(self, final_summary, marketing_analysis):
        """Save results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        domain_clean = self.base_domain.replace('.', '_').replace('www_', '')
        
        # Individual page summaries
        pages_file = f"smart_scraped_{domain_clean}_{timestamp}_pages.md"
        with open(pages_file, 'w', encoding='utf-8') as f:
            f.write(f"# Smart Web Scraper Results: {self.base_domain}\n\n")
            f.write(f"**Processed:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Total Pages:** {len(self.page_summaries)}\n")
            f.write(f"**Configuration:** {self.max_workers} workers, {self.max_pages} pages target\n\n")
            f.write("---\n\n")
            
            for i, summary in enumerate(self.page_summaries, 1):
                f.write(f"## {i}. {summary['title']}\n\n")
                f.write(f"**URL:** {summary['url']}\n")
                f.write(f"**Word Count:** {summary['word_count']:,}\n\n")
                f.write(f"### Summary\n{summary['summary']}\n\n")
                f.write("---\n\n")
        
        # Final comprehensive summary
        final_file = f"smart_scraped_{domain_clean}_{timestamp}_final.md"
        with open(final_file, 'w', encoding='utf-8') as f:
            f.write(final_summary)
        
        # Marketing analysis
        marketing_file = f"smart_scraped_{domain_clean}_{timestamp}_marketing.md"
        with open(marketing_file, 'w', encoding='utf-8') as f:
            f.write(f"# Marketing Analysis: {self.base_domain}\n\n")
            f.write(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Source:** Analysis of {len(self.page_summaries)} pages\n\n")
            f.write("---\n\n")
            f.write(marketing_analysis)
        
        print(f"\n💾 Results saved:")
        print(f"  📄 Individual pages: {pages_file}")
        print(f"  📄 Final summary: {final_file}")
        print(f"  🎯 Marketing analysis: {marketing_file}")
        
        return pages_file, final_file, marketing_file

    def run(self):
        """Main scraping execution"""
        print(f"🧠 Smart Web Scraper Starting")
        print(f"🎯 Target: {self.base_url}")
        print(f"⚙️  Config: {self.max_workers} workers, {self.max_pages} pages max")
        print(f"🤖 Robots.txt: {'✓ Loaded' if self.robots_parser else '✗ Not found'}")
        print("=" * 60)
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            consecutive_failures = 0
            max_consecutive_failures = 10
            
            while (len(self.page_summaries) < self.max_pages and 
                   consecutive_failures < max_consecutive_failures):
                
                # Submit new tasks
                while len(futures) < self.max_workers:
                    url = self.get_next_url()
                    if not url:
                        break
                    
                    future = executor.submit(self.process_url, url)
                    futures.append(future)
                
                if not futures:
                    break
                
                # Process completed futures
                completed = []
                success_this_round = 0
                
                for future in futures:
                    if future.done():
                        completed.append(future)
                        try:
                            result = future.result()
                            if result:
                                success_this_round += 1
                                consecutive_failures = 0
                                
                                # Progress update
                                with self.lock:
                                    processed = len(self.page_summaries)
                                    queue_size = len(self.priority_urls) + len(self.regular_urls)
                                
                                elapsed = time.time() - start_time
                                rate = processed / elapsed if elapsed > 0 else 0
                                
                                print(f"📊 Progress: {processed}/{self.max_pages} | Queue: {queue_size} | Rate: {rate:.1f}/sec")
                            else:
                                consecutive_failures += 1
                        except Exception as e:
                            consecutive_failures += 1
                            print(f"❌ Task failed: {e}")
                
                # Remove completed futures
                for future in completed:
                    futures.remove(future)
                
                # Adaptive behavior based on success rate
                if consecutive_failures > 5:
                    print(f"⚠️  Many failures detected. Slowing down...")
                    time.sleep(5)
                
                time.sleep(0.5)  # Small delay between rounds
        
        if not self.page_summaries:
            print("❌ No pages were successfully processed!")
            return None
        
        # Create final summary
        final_summary = self.create_final_summary()
        
        # Create marketing analysis
        marketing_analysis = self.create_marketing_analysis(final_summary)
        
        # Save results
        files = self.save_results(final_summary, marketing_analysis)
        
        # Final results
        total_time = time.time() - start_time
        self._print_final_stats(total_time)
        
        return {
            'summaries': self.page_summaries,
            'final_summary': final_summary,
            'marketing_analysis': marketing_analysis,
            'files': files,
            'stats': {
                'time': total_time,
                'pages': len(self.page_summaries),
                'words': sum(s['word_count'] for s in self.page_summaries),
                'speed': len(self.page_summaries)/total_time if total_time > 0 else 0
            }
        }
    
    def _print_final_stats(self, total_time):
        """Print comprehensive final statistics"""
        print(f"\n🎉 Smart Scraping Complete!")
        print(f"⏱️  Total time: {total_time:.1f} seconds")
        print(f"📄 Pages successfully processed: {len(self.page_summaries)}")
        print(f"🔍 Total URLs discovered: {len(self.visited)}")
        print(f"🚫 URLs filtered out: {self.stats['urls_filtered']}")
        print(f"📡 HTTP requests made: {self.stats['requests_made']}")
        print(f"❌ Error encounters: {self.stats['errors_encountered']}")
        print(f"🐌 Rate limit hits: {self.stats['rate_limit_hits']}")
        print(f"🚨 Circuit breaker trips: {self.stats['circuit_breaker_trips']}")
        
        if len(self.page_summaries) > 0:
            rate = len(self.page_summaries) / total_time
            print(f"🚀 Success rate: {rate:.2f} pages/second")
            
            success_rate = (len(self.page_summaries) / self.stats['requests_made']) * 100 if self.stats['requests_made'] > 0 else 0
            print(f"📊 Request success rate: {success_rate:.1f}%")
        
        print(f"🧠 Final adaptive delay: {self.adaptive_delay:.1f}s")

def main():
    print("🧠 Smart Web Scraper with Intelligent Validation")
    print("=" * 50)
    
    # Get configuration
    url = input("Enter website URL: ").strip()
    if not url.startswith(('http://', 'https://')):
        url = f"https://{url}"
    
    try:
        max_pages = int(input("Max pages [25]: ").strip() or "25")
        max_workers = int(input("Workers [2]: ").strip() or "2")
    except ValueError:
        max_pages, max_workers = 25, 2
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("⚠️  No OpenAI API key found - will run without AI summaries")
    
    print(f"\n🚀 Starting smart crawl with {max_workers} workers targeting {max_pages} pages")
    
    # Initialize and run scraper
    scraper = SmartWebScraper(url, api_key, max_workers, max_pages)
    results = scraper.run()
    
    if results:
        print(f"\n🎉 SUCCESS! Website analysis complete.")
        print(f"📁 Check the generated markdown files for detailed results:")
        print(f"   📄 Individual page summaries")  
        print(f"   📄 Comprehensive final summary")
        print(f"   🎯 Marketing analysis breakdown")
        
        # Show some stats about what was found
        if results.get('summaries'):
            print(f"\n📋 Sample pages found:")
            for i, page in enumerate(results['summaries'][:5], 1):
                print(f"  {i}. {page['title'][:50]}{'...' if len(page['title']) > 50 else ''}")
            
            if len(results['summaries']) > 5:
                print(f"  ... and {len(results['summaries']) - 5} more!")
                
            print(f"\n📊 Final Stats:")
            print(f"  ⏱️  Total time: {results['stats']['time']:.1f} seconds")
            print(f"  📄 Pages processed: {results['stats']['pages']}")
            print(f"  📝 Total words: {results['stats']['words']:,}")
            print(f"  🚀 Processing speed: {results['stats']['speed']:.2f} pages/second")
        
    else:
        print("\n❌ No pages were successfully processed")
        print("💡 Tips:")
        print("   - Make sure the website is accessible")
        print("   - Try with fewer workers (1) if getting many errors")
        print("   - Some sites have aggressive bot protection")
        print("   - Check your internet connection")

if __name__ == "__main__":
    main()

