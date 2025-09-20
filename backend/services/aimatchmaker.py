import re
import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Tuple, Optional
import logging
from rapidfuzz import process, fuzz

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    pass

class AIResumeEvaluator:
    """Advanced AI-powered resume evaluation system with enhanced skill extraction"""
    
    def __init__(self):
        # Initialize sentence transformer
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_available = True
        except Exception as e:
            print(f"Warning: Sentence transformer not available: {e}")
            self.embedding_available = False
        
        # Comprehensive skill database
        self.skill_categories = {
            'programming_languages': [
                'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go',
                'rust', 'swift', 'kotlin', 'scala', 'ruby', 'php', 'r'
            ],
            'web_technologies': [
                'html', 'css', 'react', 'angular', 'vue.js', 'node.js', 'express.js',
                'next.js', 'nuxt.js', 'svelte', 'bootstrap', 'tailwind css'
            ],
            'frameworks': [
                'django', 'flask', 'fastapi', 'spring boot', 'spring', 'laravel',
                'rails', 'asp.net', 'hibernate', 'struts'
            ],
            'databases': [
                'mysql', 'postgresql', 'mongodb', 'redis', 'cassandra', 'sqlite',
                'oracle', 'sql server', 'elasticsearch', 'neo4j'
            ],
            'cloud_platforms': [
                'aws', 'azure', 'gcp', 'google cloud platform', 'heroku',
                'digitalocean', 'linode', 'ibm cloud'
            ],
            'devops_tools': [
                'docker', 'kubernetes', 'jenkins', 'gitlab ci', 'github actions',
                'terraform', 'ansible', 'puppet', 'chef', 'vagrant'
            ],
            'data_science': [
                'machine learning', 'deep learning', 'tensorflow', 'pytorch',
                'keras', 'scikit-learn', 'pandas', 'numpy', 'matplotlib',
                'seaborn', 'jupyter', 'tableau', 'power bi'
            ],
            'mobile_development': [
                'android', 'ios', 'react native', 'flutter', 'xamarin',
                'ionic', 'cordova', 'swift', 'kotlin'
            ]
        }
        
        # Flatten all skills for easy access
        self.all_skills = []
        for category, skills in self.skill_categories.items():
            self.all_skills.extend(skills)
        
        # Common experience patterns
        self.experience_patterns = [
            r'(\d+)\s*\+?\s*years?\s+(?:of\s+)?experience',
            r'(\d+)\s*\+?\s*yrs?\s+(?:of\s+)?experience',
            r'experience\s*:?\s*(\d+)\s*\+?\s*years?',
            r'(\d+)\s*\+?\s*years?\s+(?:in|with)\s+\w+',
            r'(\d+)\s*\+?\s*year\s+experience'
        ]
    
    def evaluate(self, resume_text: str, jd_text: str) -> Dict:
        """Main evaluation function"""
        try:
            # Extract skills from both texts
            resume_skills = self._extract_skills(resume_text)
            jd_skills = self._extract_skills(jd_text)
            
            # Calculate different types of matches
            skill_match_score = self._calculate_skill_match(resume_skills, jd_skills)
            semantic_similarity = self._calculate_semantic_similarity(resume_text, jd_text)
            
            # Extract experience
            resume_experience = self._extract_experience_years(resume_text)
            required_experience = self._extract_experience_years(jd_text)
            
            # Calculate overall score with weights
            overall_score = self._calculate_weighted_score(
                skill_match_score, semantic_similarity, resume_experience, required_experience
            )
            
            # Generate insights
            skill_gaps = self._find_skill_gaps(resume_skills, jd_skills)
            recommendations = self._generate_recommendations(skill_gaps, resume_skills)
            
            return {
                'overall_score': round(overall_score, 2),
                'skill_match_score': round(skill_match_score, 2),
                'semantic_similarity': round(semantic_similarity, 2),
                'resume_skills': resume_skills,
                'jd_skills': jd_skills,
                'matched_skills': list(set(resume_skills) & set(jd_skills)),
                'skill_gaps': skill_gaps,
                'experience_years': resume_experience,
                'required_experience': required_experience,
                'recommendations': recommendations,
                'feedback': self._generate_feedback(overall_score, skill_gaps, resume_skills)
            }
        
        except Exception as e:
            return {
                'error': f"Evaluation failed: {str(e)}",
                'overall_score': 0
            }
    
    def _extract_skills(self, text: str) -> List[str]:
        """Enhanced skill extraction with fuzzy matching"""
        text_lower = text.lower()
        found_skills = set()
        
        # Direct matching
        for skill in self.all_skills:
            if skill.lower() in text_lower:
                found_skills.add(skill)
        
        # Tokenize for fuzzy matching
        tokens = re.findall(r'\b[a-z0-9\+#\.\-]{2,}\b', text_lower)
        tokens = set(tokens)  # Get unique tokens
        
        # Fuzzy matching for OCR errors
        for token in tokens:
            matches = process.extract(token, self.all_skills, scorer=fuzz.ratio, limit=3)
            for match_skill, score, _ in matches:
                if score >= 80:  # 80% similarity threshold
                    found_skills.add(match_skill)
        
        # Pattern-based extraction for variations
        skill_patterns = {
            'node': r'\bnode\.?js\b',
            'react': r'\breact\.?js\b',
            'vue': r'\bvue\.?js\b',
            'c++': r'\bc\+\+\b',
            'c#': r'\bc#\b',
            'asp.net': r'\basp\.net\b',
            'machine learning': r'\b(?:machine learning|ml)\b',
            'deep learning': r'\b(?:deep learning|dl)\b'
        }
        
        for skill, pattern in skill_patterns.items():
            if re.search(pattern, text_lower):
                found_skills.add(skill)
        
        return list(found_skills)
    
    # The rest of the methods remain the same...
    def _calculate_skill_match(self, resume_skills: List[str], jd_skills: List[str]) -> float:
        """Calculate skill matching percentage"""
        if not jd_skills:
            return 0.0
        
        resume_skills_lower = [skill.lower() for skill in resume_skills]
        jd_skills_lower = [skill.lower() for skill in jd_skills]
        
        matched_skills = set(resume_skills_lower) & set(jd_skills_lower)
        return (len(matched_skills) / len(jd_skills_lower)) * 100
    
    def _calculate_semantic_similarity(self, resume_text: str, jd_text: str) -> float:
        """Calculate semantic similarity between resume and JD"""
        if self.embedding_available:
            try:
                # Use sentence transformers for semantic similarity
                embeddings = self.sentence_model.encode([resume_text, jd_text])
                similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                return similarity * 100
            except Exception as e:
                print(f"Sentence transformer error: {e}")
        
        # Fallback to TF-IDF similarity
        try:
            vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=1000,
                ngram_range=(1, 2)
            )
            
            tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity * 100
        except:
            return 50.0  # Default fallback
    
    def _extract_experience_years(self, text: str) -> int:
        """Extract years of experience from text"""
        text_lower = text.lower()
        
        for pattern in self.experience_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                try:
                    years = int(matches[0])
                    return min(years, 20)  # Cap at 20 years for sanity
                except ValueError:
                    continue
        
        return 0
    
    def _calculate_weighted_score(self, skill_score: float, semantic_score: float, 
                                 resume_exp: int, required_exp: int) -> float:
        """Calculate weighted overall score"""
        # Experience score
        if required_exp == 0:
            exp_score = 100
        else:
            exp_score = min(100, (resume_exp / required_exp) * 100)
        
        # Weighted combination
        weights = {
            'skills': 0.5,
            'semantic': 0.3,
            'experience': 0.2
        }
        
        overall_score = (
            skill_score * weights['skills'] +
            semantic_score * weights['semantic'] +
            exp_score * weights['experience']
        )
        
        return min(100, overall_score)
    
    def _find_skill_gaps(self, resume_skills: List[str], jd_skills: List[str]) -> List[str]:
        """Find missing skills"""
        resume_skills_lower = [skill.lower() for skill in resume_skills]
        jd_skills_lower = [skill.lower() for skill in jd_skills]
        
        gaps = [skill for skill in jd_skills_lower if skill not in resume_skills_lower]
        return gaps
    
    def _generate_recommendations(self, skill_gaps: List[str], current_skills: List[str]) -> List[str]:
        """Generate learning recommendations"""
        recommendations = []
        
        if skill_gaps:
            # Prioritize gaps by category
            priority_categories = ['programming_languages', 'web_technologies', 'cloud_platforms']
            
            for category in priority_categories:
                category_skills = [s.lower() for s in self.skill_categories[category]]
                category_gaps = [gap for gap in skill_gaps if gap in category_skills]
                
                if category_gaps:
                    recommendations.append(f"Focus on {category.replace('_', ' ')}: {', '.join(category_gaps[:3])}")
        
        # Add general recommendations
        if len(current_skills) < 5:
            recommendations.append("Build a stronger technical skill foundation")
        
        return recommendations[:5]  # Limit to top 5
    
    def _generate_feedback(self, score: float, gaps: List[str], skills: List[str]) -> str:
        """Generate personalized feedback"""
        if score >= 80:
            feedback = "🎉 Excellent match! You have most of the required qualifications."
        elif score >= 65:
            feedback = "👍 Good match with some areas for improvement."
        elif score >= 50:
            feedback = "⚡ Moderate match. Focus on building key skills."
        else:
            feedback = "📚 Significant skill development needed for this role."
        
        if gaps:
            feedback += f" Priority skills to learn: {', '.join(gaps[:3])}."
        
        if len(skills) >= 8:
            feedback += " Your diverse skill set is impressive!"
        
        return feedback
    
    def analyze_job_description(self, jd_text: str) -> Dict:
        """Analyze job description to extract key requirements"""
        skills = self._extract_skills(jd_text)
        experience = self._extract_experience_years(jd_text)
        
        # Categorize skills
        categorized_skills = {}
        for category, category_skills in self.skill_categories.items():
            found_in_category = [skill for skill in skills if skill.lower() in [s.lower() for s in category_skills]]
            if found_in_category:
                categorized_skills[category] = found_in_category
        
        return {
            'total_skills': len(skills),
            'all_skills': skills,
            'categorized_skills': categorized_skills,
            'required_experience': experience,
            'complexity_score': min(100, len(skills) * 10 + experience * 5)
        }