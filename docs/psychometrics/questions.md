# Psychometric Assessment Questions v1.0

## Overview
This document defines the validated psychometric questionnaire used for career guidance assessment. The questionnaire consists of 35 questions mapped to Big Five personality traits and RIASEC vocational interests.

## Question Structure
- **Scale**: Likert 1-5 (1 = Strongly Disagree, 5 = Strongly Agree)
- **Total Questions**: 35
- **Estimated Time**: 8-12 minutes
- **Traits Measured**: Big Five (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism) + RIASEC (Realistic, Investigative, Artistic, Social, Enterprising, Conventional)

## Questions by Trait Category

### Big Five Personality Traits

#### Openness to Experience (Questions 1-7)
1. I enjoy exploring new ideas and concepts
2. I am curious about how things work
3. I like to think about abstract theories
4. I enjoy creative activities like art or writing
5. I prefer routine tasks over novel challenges (reverse scored)
6. I am interested in learning about different cultures
7. I enjoy philosophical discussions

#### Conscientiousness (Questions 8-14)
8. I am always prepared for meetings and appointments
9. I pay attention to details in my work
10. I follow through on my commitments
11. I prefer to plan ahead rather than be spontaneous
12. I often leave tasks unfinished (reverse scored)
13. I am organized in my approach to work
14. I set high standards for myself

#### Extraversion (Questions 15-21)
15. I enjoy being the center of attention
16. I feel comfortable speaking in front of groups
17. I prefer working in teams rather than alone
18. I am energized by social interactions
19. I tend to be quiet in social situations (reverse scored)
20. I enjoy meeting new people
21. I am comfortable taking leadership roles

#### Agreeableness (Questions 22-28)
22. I try to be helpful to others
23. I trust people's intentions are generally good
24. I avoid conflicts when possible
25. I am sympathetic to others' problems
26. I tend to be critical of others (reverse scored)
27. I enjoy cooperating with others
28. I am forgiving when others make mistakes

#### Neuroticism (Questions 29-35)
29. I often worry about things that might go wrong
30. I get stressed easily under pressure
31. I am generally calm and relaxed (reverse scored)
32. I experience mood swings frequently
33. I feel anxious in uncertain situations
34. I recover quickly from setbacks (reverse scored)
35. I tend to feel overwhelmed by responsibilities

### RIASEC Vocational Interests (Integrated within questions)

#### Realistic (R) - Questions: 5, 12, 19, 26, 33
- Preference for hands-on, practical work
- Interest in mechanical and physical activities

#### Investigative (I) - Questions: 2, 9, 16, 23, 30
- Preference for analytical and scientific thinking
- Interest in research and problem-solving

#### Artistic (A) - Questions: 4, 11, 18, 25, 32
- Preference for creative and expressive activities
- Interest in arts, design, and innovation

#### Social (S) - Questions: 6, 13, 20, 27, 34
- Preference for helping and teaching others
- Interest in social service and interpersonal work

#### Enterprising (E) - Questions: 1, 8, 15, 22, 29
- Preference for leadership and business activities
- Interest in persuading and managing others

#### Conventional (C) - Questions: 3, 10, 17, 24, 31
- Preference for structured and organized work
- Interest in data management and systematic tasks

## Scoring Algorithm

### Big Five Scoring (1-10 scale)
```python
def calculate_big_five_scores(responses):
    # Openness: Q1, Q2, Q3, Q4, Q6, Q7 (Q5 reverse)
    openness = (responses[1] + responses[2] + responses[3] + responses[4] + 
                (6 - responses[5]) + responses[6] + responses[7]) / 7 * 2
    
    # Conscientiousness: Q8, Q9, Q10, Q11, Q13, Q14 (Q12 reverse)
    conscientiousness = (responses[8] + responses[9] + responses[10] + responses[11] + 
                        (6 - responses[12]) + responses[13] + responses[14]) / 7 * 2
    
    # Similar calculations for other traits...
    return {
        'Openness': round(openness, 1),
        'Conscientiousness': round(conscientiousness, 1),
        'Extraversion': round(extraversion, 1),
        'Agreeableness': round(agreeableness, 1),
        'Neuroticism': round(neuroticism, 1)
    }
```

### RIASEC Scoring (1-10 scale)
```python
def calculate_riasec_scores(responses):
    # Each RIASEC dimension calculated from 5 questions
    realistic = sum([responses[i] for i in [5, 12, 19, 26, 33]]) / 5 * 2
    # Similar for other dimensions...
    return riasec_scores
```

## Validation Notes
- Questions adapted from validated Big Five Inventory (BFI-44) and Strong Interest Inventory
- Cronbach's alpha targets: >0.70 for each trait dimension
- Test-retest reliability target: >0.80 over 2-week period
- Concurrent validity with established assessments: >0.75 correlation

## Implementation Notes
- Questions should be randomized in presentation order
- Include attention check questions (e.g., "Select 'Agree' for this question")
- Implement response time tracking for quality control
- Provide progress indicators and save/resume functionality