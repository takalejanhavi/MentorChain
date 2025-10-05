// Trait calculation and report generation service

const calculateTraitScores = (responses) => {
  // Convert responses array to object for easier access
  const responseMap = {};
  responses.forEach(r => {
    responseMap[r.question_id] = r.response;
  });

  // Big Five trait calculations (questions mapped to traits)
  const big_five = {
    // Openness: Q1, Q2, Q3, Q4, Q6, Q7 (Q5 reverse scored)
    Openness: calculateTraitScore([1, 2, 3, 4, 6, 7], [5], responseMap),
    
    // Conscientiousness: Q8, Q9, Q10, Q11, Q13, Q14 (Q12 reverse scored)
    Conscientiousness: calculateTraitScore([8, 9, 10, 11, 13, 14], [12], responseMap),
    
    // Extraversion: Q15, Q16, Q17, Q18, Q20, Q21 (Q19 reverse scored)
    Extraversion: calculateTraitScore([15, 16, 17, 18, 20, 21], [19], responseMap),
    
    // Agreeableness: Q22, Q23, Q24, Q25, Q27, Q28 (Q26 reverse scored)
    Agreeableness: calculateTraitScore([22, 23, 24, 25, 27, 28], [26], responseMap),
    
    // Neuroticism: Q29, Q30, Q32, Q35 (Q31, Q34 reverse scored)
    Neuroticism: calculateTraitScore([29, 30, 32, 35], [31, 34], responseMap)
  };

  // RIASEC interest calculations (distributed across questions)
  const riasec = {
    // Realistic: Questions focusing on hands-on, practical work
    Realistic: calculateTraitScore([5, 12, 19, 26, 33], [], responseMap),
    
    // Investigative: Questions focusing on analytical thinking
    Investigative: calculateTraitScore([2, 9, 16, 23, 30], [], responseMap),
    
    // Artistic: Questions focusing on creativity
    Artistic: calculateTraitScore([4, 11, 18, 25, 32], [], responseMap),
    
    // Social: Questions focusing on helping others
    Social: calculateTraitScore([6, 13, 20, 27, 34], [], responseMap),
    
    // Enterprising: Questions focusing on leadership
    Enterprising: calculateTraitScore([1, 8, 15, 22, 29], [], responseMap),
    
    // Conventional: Questions focusing on structure
    Conventional: calculateTraitScore([3, 10, 17, 24, 31], [], responseMap)
  };

  return { big_five, riasec };
};

const calculateTraitScore = (positiveQuestions, reverseQuestions, responseMap) => {
  let total = 0;
  let count = 0;

  // Add positive questions
  positiveQuestions.forEach(qId => {
    if (responseMap[qId]) {
      total += responseMap[qId];
      count++;
    }
  });

  // Add reverse-scored questions
  reverseQuestions.forEach(qId => {
    if (responseMap[qId]) {
      total += (6 - responseMap[qId]); // Reverse score (1->5, 2->4, 3->3, 4->2, 5->1)
      count++;
    }
  });

  // Convert to 1-10 scale
  const average = count > 0 ? total / count : 3; // Default to neutral if no responses
  return Math.round((average - 1) * (10 / 4) * 10) / 10; // Scale 1-5 to 1-10, round to 1 decimal
};

const generateCareerReport = async (predictions, traitScores) => {
  const topCareer = predictions.predictions[0];
  const { big_five, riasec } = traitScores;

  // Generate summary
  const summary = generateSummary(topCareer, big_five, riasec);
  
  // Generate strengths based on high trait scores
  const strengths = generateStrengths(big_five, riasec);
  
  // Generate development areas based on lower trait scores
  const developmentAreas = generateDevelopmentAreas(big_five, riasec);
  
  // Generate action steps based on top career predictions
  const actionSteps = generateActionSteps(predictions.predictions);
  
  // Generate learning resources
  const resources = generateResources(predictions.predictions);

  return {
    summary,
    strengths,
    development_areas: developmentAreas,
    action_steps: actionSteps,
    resources
  };
};

const generateSummary = (topCareer, bigFive, riasec) => {
  const careerField = topCareer.career.split(' → ')[0];
  const probability = (topCareer.probability * 100).toFixed(1);
  
  // Find dominant personality trait
  const dominantTrait = Object.entries(bigFive).reduce((a, b) => 
    bigFive[a[0]] > bigFive[b[0]] ? a : b
  )[0];
  
  // Find dominant interest
  const dominantInterest = Object.entries(riasec).reduce((a, b) => 
    riasec[a[0]] > riasec[b[0]] ? a : b
  )[0];

  return `Based on your psychometric assessment, you show a ${probability}% compatibility with ${careerField} careers. Your personality profile indicates high ${dominantTrait.toLowerCase()} and strong ${dominantInterest.toLowerCase()} interests, which align well with this career path. This assessment analyzed your responses across 35 validated questions measuring personality traits and vocational interests to provide personalized career guidance.`;
};

const generateStrengths = (bigFive, riasec) => {
  const strengths = [];
  
  // Big Five strengths (scores > 7)
  if (bigFive.Openness > 7) {
    strengths.push("High creativity and openness to new experiences - you thrive in innovative environments");
  }
  if (bigFive.Conscientiousness > 7) {
    strengths.push("Strong organizational skills and attention to detail - you excel at planning and execution");
  }
  if (bigFive.Extraversion > 7) {
    strengths.push("Excellent interpersonal skills and leadership potential - you energize teams and communicate effectively");
  }
  if (bigFive.Agreeableness > 7) {
    strengths.push("Natural collaboration abilities and empathy - you build strong relationships and work well with others");
  }
  if (bigFive.Neuroticism < 4) { // Low neuroticism is a strength
    strengths.push("Emotional stability and resilience under pressure - you maintain composure in challenging situations");
  }

  // RIASEC strengths (scores > 7)
  if (riasec.Realistic > 7) {
    strengths.push("Practical problem-solving skills and hands-on approach to work");
  }
  if (riasec.Investigative > 7) {
    strengths.push("Analytical thinking and research capabilities - you excel at systematic investigation");
  }
  if (riasec.Artistic > 7) {
    strengths.push("Creative expression and innovative thinking - you bring fresh perspectives to challenges");
  }
  if (riasec.Social > 7) {
    strengths.push("Strong desire to help others and natural teaching abilities");
  }
  if (riasec.Enterprising > 7) {
    strengths.push("Leadership potential and business acumen - you can influence and motivate others");
  }
  if (riasec.Conventional > 7) {
    strengths.push("Systematic approach and attention to procedures - you excel in structured environments");
  }

  // Ensure we have at least 3 strengths
  while (strengths.length < 3) {
    const fallbackStrengths = [
      "Balanced personality profile that adapts well to various work environments",
      "Consistent response pattern indicating self-awareness and honest self-assessment",
      "Diverse interests that provide flexibility in career choices"
    ];
    strengths.push(fallbackStrengths[strengths.length]);
  }

  return strengths.slice(0, 3);
};

const generateDevelopmentAreas = (bigFive, riasec) => {
  const areas = [];
  
  // Big Five development areas (scores < 4)
  if (bigFive.Openness < 4) {
    areas.push("Consider expanding comfort zone by exploring new ideas and approaches to work");
  }
  if (bigFive.Conscientiousness < 4) {
    areas.push("Focus on developing organizational systems and time management skills");
  }
  if (bigFive.Extraversion < 4) {
    areas.push("Practice public speaking and networking to build confidence in social professional settings");
  }
  if (bigFive.Agreeableness < 4) {
    areas.push("Work on collaborative skills and consider others' perspectives in decision-making");
  }
  if (bigFive.Neuroticism > 7) {
    areas.push("Develop stress management techniques and emotional regulation strategies");
  }

  // General development areas if no specific low scores
  if (areas.length === 0) {
    areas.push("Continue developing technical skills relevant to your chosen career field");
    areas.push("Seek opportunities to expand your professional network and industry knowledge");
    areas.push("Consider developing complementary skills that enhance your primary strengths");
  }

  return areas.slice(0, 3);
};

const generateActionSteps = (predictions) => {
  const steps = [];
  const topCareer = predictions[0].career;
  const careerField = topCareer.split(' → ')[0];
  const subfield = topCareer.split(' → ')[1];

  // Career-specific action steps
  const careerActions = {
    'Doctor': [
      `Research medical specializations in ${subfield} and required educational pathways`,
      'Volunteer at healthcare facilities to gain hands-on experience',
      'Connect with medical professionals through informational interviews'
    ],
    'Engineer': [
      `Explore ${subfield} engineering programs and certification requirements`,
      'Build a portfolio of technical projects demonstrating problem-solving skills',
      'Join professional engineering societies and attend industry conferences'
    ],
    'IT Professional': [
      `Learn specific technologies used in ${subfield} through online courses`,
      'Contribute to open-source projects to build your technical portfolio',
      'Obtain relevant certifications and stay updated with industry trends'
    ],
    'Designer': [
      `Create a portfolio showcasing ${subfield} design work`,
      'Study current design trends and user experience principles',
      'Network with design professionals and seek mentorship opportunities'
    ],
    'Business': [
      `Research ${subfield} business roles and required qualifications`,
      'Develop analytical and communication skills through relevant coursework',
      'Seek internships or entry-level positions in business environments'
    ]
  };

  const specificActions = careerActions[careerField] || [
    `Research educational requirements and career paths in ${careerField}`,
    'Network with professionals in your field of interest',
    'Gain relevant experience through internships, volunteering, or projects'
  ];

  return specificActions;
};

const generateResources = (predictions) => {
  const resources = [];
  const topCareer = predictions[0].career;
  const careerField = topCareer.split(' → ')[0];

  // Career-specific resources
  const careerResources = {
    'Doctor': [
      { title: 'Association of American Medical Colleges (AAMC)', url: 'https://www.aamc.org' },
      { title: 'Medscape Career Center', url: 'https://www.medscape.com/career' }
    ],
    'Engineer': [
      { title: 'National Society of Professional Engineers', url: 'https://www.nspe.org' },
      { title: 'IEEE Career Center', url: 'https://careers.ieee.org' }
    ],
    'IT Professional': [
      { title: 'CompTIA Career Resources', url: 'https://www.comptia.org/careers' },
      { title: 'Stack Overflow Developer Survey', url: 'https://insights.stackoverflow.com/survey' }
    ],
    'Designer': [
      { title: 'AIGA Design Career Resources', url: 'https://www.aiga.org/career-resources' },
      { title: 'Interaction Design Foundation', url: 'https://www.interaction-design.org' }
    ],
    'Business': [
      { title: 'Harvard Business Review Career Advice', url: 'https://hbr.org/topic/career-planning' },
      { title: 'Bureau of Labor Statistics Career Guide', url: 'https://www.bls.gov/careerguide' }
    ]
  };

  const specificResources = careerResources[careerField] || [
    { title: 'O*NET Interest Profiler', url: 'https://www.mynextmove.org/explore/ip' },
    { title: 'Bureau of Labor Statistics Occupational Outlook', url: 'https://www.bls.gov/ooh' }
  ];

  return specificResources;
};

module.exports = {
  calculateTraitScores,
  generateCareerReport
};