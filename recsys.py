import numpy as np # type: ignore
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum
import pandas as pd # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore

# Enums for various classifications
class PersonalityType(Enum):
    INTROVERT = "introvert"
    EXTROVERT = "extrovert"
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    STRUCTURED = "structured"
    FLEXIBLE = "flexible"

class WorkEnvironment(Enum):
    REMOTE = "remote"
    OFFICE = "office"
    HYBRID = "hybrid"
    STARTUP = "startup"
    CORPORATE = "corporate"

# Data classes for different aspects of the recommendation system
@dataclass
class PersonalityProfile:
    big_five_scores: Dict[str, float]  # openness, conscientiousness, extraversion, agreeableness, neuroticism
    mbti_type: str
    work_values: List[str]
    preferred_environment: List[WorkEnvironment]
    personality_types: List[PersonalityType]

@dataclass
class SkillProfile:
    technical_skills: List[str]
    soft_skills: List[str]
    experience_level: int
    certifications: List[str]

@dataclass
class EmotionalProfile:
    stress_tolerance: float
    work_life_balance_preference: float
    growth_mindset: float
    emotional_stability: float
    job_satisfaction_history: List[float]

@dataclass
class User:
    id: str
    personality_profile: PersonalityProfile
    skill_profile: SkillProfile
    emotional_profile: EmotionalProfile
    career_goals: List[str]

@dataclass
class CareerOpportunity:
    id: str
    title: str
    company: str
    required_skills: List[str]
    work_environment: WorkEnvironment
    culture_values: List[str]
    personality_fit: List[PersonalityType]
    growth_opportunities: List[str]
    stress_level: float

class FitPathRecommender:
    def _init_(self):
        self.users = {}
        self.opportunities = {}
        self.learning_resources = {}
        
    def add_user(self, user: User):
        """Add or update a user in the system"""
        self.users[user.id] = user
        
    def add_opportunity(self, opportunity: CareerOpportunity):
        """Add or update a career opportunity"""
        self.opportunities[opportunity.id] = opportunity
        
    def calculate_personality_match(self, user: User, opportunity: CareerOpportunity) -> float:
        """Calculate personality compatibility score"""
        # Convert personality types to numerical vectors
        user_personality = set(user.personality_profile.personality_types)
        opp_personality = set(opportunity.personality_fit)
        
        # Calculate Jaccard similarity for personality match
        intersection = len(user_personality.intersection(opp_personality))
        union = len(user_personality.union(opp_personality))
        personality_score = intersection / union if union > 0 else 0
        
        # Consider work environment compatibility
        env_match = opportunity.work_environment in user.personality_profile.preferred_environment
        env_score = 1.0 if env_match else 0.0
        
        return 0.7 * personality_score + 0.3 * env_score
    
    def calculate_emotional_fit(self, user: User, opportunity: CareerOpportunity) -> float:
        """Calculate emotional compatibility score"""
        # Compare stress tolerance with job stress level
        stress_compatibility = 1 - abs(user.emotional_profile.stress_tolerance - opportunity.stress_level)
        
        # Consider emotional stability and growth mindset
        emotional_health = (user.emotional_profile.emotional_stability + 
                          user.emotional_profile.growth_mindset) / 2
        
        return 0.6 * stress_compatibility + 0.4 * emotional_health
    
    def calculate_skill_match(self, user: User, opportunity: CareerOpportunity) -> float:
        """Calculate skill compatibility score"""
        required_skills = set(opportunity.required_skills)
        user_skills = set(user.skill_profile.technical_skills + user.skill_profile.soft_skills)
        
        # Calculate skill overlap
        matching_skills = len(required_skills.intersection(user_skills))
        total_required = len(required_skills)
        
        return matching_skills / total_required if total_required > 0 else 0
    
    def get_recommendations(self, user_id: str, limit: int = 5) -> List[Dict]:
        """Get personalized career recommendations for a user"""
        user = self.users.get(user_id)
        if not user:
            raise ValueError("User not found")
            
        recommendations = []
        for opp_id, opportunity in self.opportunities.items():
            # Calculate different aspects of compatibility
            personality_score = self.calculate_personality_match(user, opportunity)
            emotional_score = self.calculate_emotional_fit(user, opportunity)
            skill_score = self.calculate_skill_match(user, opportunity)
            
            # Weighted total score
            total_score = (0.4 * personality_score + 
                         0.3 * emotional_score + 
                         0.3 * skill_score)
            
            recommendations.append({
                'opportunity_id': opp_id,
                'title': opportunity.title,
                'company': opportunity.company,
                'total_score': total_score,
                'personality_match': personality_score,
                'emotional_fit': emotional_score,
                'skill_match': skill_score,
                'growth_opportunities': opportunity.growth_opportunities,
                'work_environment': opportunity.work_environment.value
            })
        
        # Sort by total score and return top recommendations
        recommendations.sort(key=lambda x: x['total_score'], reverse=True)
        return recommendations[:limit]
    
    def track_emotional_wellbeing(self, user_id: str, satisfaction_score: float) -> None:
        """Track and update user's emotional well-being"""
        user = self.users.get(user_id)
        if user:
            user.emotional_profile.job_satisfaction_history.append(satisfaction_score)
            # If satisfaction is consistently low, trigger re-evaluation
            if len(user.emotional_profile.job_satisfaction_history) >= 3:
                recent_satisfaction = user.emotional_profile.job_satisfaction_history[-3:]
                if sum(recent_satisfaction) / 3 < 0.6:
                    return self.get_recommendations(user_id, limit=3)
        
    def suggest_learning_path(self, user_id: str, opportunity_id: str) -> List[str]:
        """Generate personalized learning recommendations"""
        user = self.users.get(user_id)
        opportunity = self.opportunities.get(opportunity_id)
        
        if not user or not opportunity:
            raise ValueError("User or opportunity not found")
            
        # Identify skill gaps
        required_skills = set(opportunity.required_skills)
        user_skills = set(user.skill_profile.technical_skills + user.skill_profile.soft_skills)
        skill_gaps = required_skills - user_skills
        
        # Return relevant learning resources
        learning_path = []
        for skill in skill_gaps:
            if skill in self.learning_resources:
                learning_path.append(self.learning_resources[skill])
        
        return learning_path

def create_sample_recommender():
    """Create a sample recommender with some test data"""
    recommender = FitPathRecommender()
    
    # Add sample user
    user = User(
        id="user1",
        personality_profile=PersonalityProfile(
            big_five_scores={
                "openness": 0.8,
                "conscientiousness": 0.7,
                "extraversion": 0.6,
                "agreeableness": 0.75,
                "neuroticism": 0.3
            },
            mbti_type="INFJ",
            work_values=["creativity", "innovation", "work-life-balance"],
            preferred_environment=[WorkEnvironment.REMOTE, WorkEnvironment.HYBRID],
            personality_types=[PersonalityType.CREATIVE, PersonalityType.INTROVERT]
        ),
        skill_profile=SkillProfile(
            technical_skills=["python", "data analysis", "machine learning"],
            soft_skills=["communication", "problem-solving"],
            experience_level=3,
            certifications=["AWS Certified"]
        ),
        emotional_profile=EmotionalProfile(
            stress_tolerance=0.7,
            work_life_balance_preference=0.8,
            growth_mindset=0.9,
            emotional_stability=0.8,
            job_satisfaction_history=[0.8, 0.7, 0.9]
        ),
        career_goals=["data scientist", "team lead"]
    )
    recommender.add_user(user)
    
    # Add sample opportunity
    opportunity = CareerOpportunity(
        id="job1",
        title="Senior Data Scientist",
        company="Tech Corp",
        required_skills=["python", "machine learning", "leadership"],
        work_environment=WorkEnvironment.HYBRID,
        culture_values=["innovation", "work-life-balance"],
        personality_fit=[PersonalityType.ANALYTICAL, PersonalityType.CREATIVE],
        growth_opportunities=["management track", "research projects"],
        stress_level=0.6
    )
    recommender.add_opportunity(opportunity)
    
    return recommender

# Example usage
if __name__ == "_main_":
    recommender = create_sample_recommender()
    recommendations = recommender.get_recommendations("user1")
    print("Career Recommendations:")
    for rec in recommendations:
        print(f"\nTitle: {rec['title']}")
        print(f"Company: {rec['company']}")
        print(f"Total Match Score: {rec['total_score']:.2f}")
        print(f"Personality Match: {rec['personality_match']:.2f}")
        print(f"Emotional Fit: {rec['emotional_fit']:.2f}")
        print(f"Skill Match: {rec['skill_match']:.2f}")
