<style>
  :root {
    --primary: #00d9ff;
    --secondary: #ff006e;
    --tertiary: #ffbe0b;
    --dark: #0a0e27;
    --light: #f0f7ff;
  }

  * {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }

  body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(135deg, #0a0e27 0%, #16213e 50%, #0f3460 100%);
    color: #f0f7ff;
    overflow-x: hidden;
  }

  .container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 40px 20px;
  }

  /* HERO SECTION WITH ANIMATED BACKGROUND */
  .hero {
    position: relative;
    min-height: 600px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    overflow: hidden;
    border-radius: 20px;
    background: radial-gradient(ellipse at center, rgba(0, 217, 255, 0.1) 0%, transparent 70%);
    border: 2px solid rgba(0, 217, 255, 0.3);
    margin-bottom: 60px;
    box-shadow: 0 0 60px rgba(0, 217, 255, 0.15), inset 0 0 60px rgba(0, 217, 255, 0.05);
  }

  /* Animated background shapes */
  .animated-bg {
    position: absolute;
    width: 300px;
    height: 300px;
    border-radius: 50%;
    mix-blend-mode: screen;
    opacity: 0.3;
    animation: float 8s ease-in-out infinite;
  }

  .shape-1 {
    top: -100px;
    left: -100px;
    background: radial-gradient(circle, #00d9ff, transparent);
    animation: float 8s ease-in-out infinite;
  }

  .shape-2 {
    top: 50%;
    right: -150px;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, #ff006e, transparent);
    animation: float 10s ease-in-out infinite;
    animation-delay: -2s;
  }

  .shape-3 {
    bottom: -100px;
    left: 10%;
    background: radial-gradient(circle, #ffbe0b, transparent);
    animation: float 12s ease-in-out infinite;
    animation-delay: -4s;
  }

  @keyframes float {
    0%, 100% { transform: translate(0, 0) scale(1); }
    50% { transform: translate(30px, -50px) scale(1.1); }
  }

  /* GLOWING TITLE */
  .title-wrapper {
    position: relative;
    z-index: 10;
    margin-bottom: 30px;
  }

  .glowing-title {
    font-size: 72px;
    font-weight: 900;
    letter-spacing: -2px;
    margin: 0;
    background: linear-gradient(45deg, #00d9ff, #ffbe0b, #ff006e, #00d9ff);
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: gradientShift 8s ease infinite, glow 3s ease-in-out infinite;
    text-shadow: 0 0 30px rgba(0, 217, 255, 0.5);
    filter: drop-shadow(0 0 20px rgba(0, 217, 255, 0.4));
  }

  @keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
  }

  @keyframes glow {
    0%, 100% { 
      filter: drop-shadow(0 0 20px rgba(0, 217, 255, 0.4)) drop-shadow(0 0 40px rgba(0, 217, 255, 0.2));
    }
    50% { 
      filter: drop-shadow(0 0 30px rgba(0, 217, 255, 0.6)) drop-shadow(0 0 60px rgba(0, 217, 255, 0.3));
    }
  }

  /* ANIMATED SUBTITLE */
  .subtitle {
    font-size: 28px;
    font-weight: 300;
    color: #00d9ff;
    margin-bottom: 40px;
    letter-spacing: 2px;
    position: relative;
    z-index: 10;
    animation: slideInUp 1s ease-out 0.3s both;
  }

  @keyframes slideInUp {
    from {
      opacity: 0;
      transform: translateY(50px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }

  /* TYPING EFFECT */
  .typing-effect {
    position: relative;
    z-index: 10;
    font-size: 20px;
    color: #ffbe0b;
    height: 30px;
    font-weight: 600;
    letter-spacing: 1px;
  }

  .typing-effect::after {
    content: '|';
    animation: blink 0.7s infinite;
    margin-left: 5px;
  }

  @keyframes blink {
    0%, 49% { opacity: 1; }
    50%, 100% { opacity: 0; }
  }

  /* CTA BUTTON WITH HOVER ANIMATION */
  .cta-button {
    position: relative;
    z-index: 10;
    padding: 16px 48px;
    font-size: 18px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    border: 2px solid #00d9ff;
    background: transparent;
    color: #00d9ff;
    cursor: pointer;
    border-radius: 50px;
    overflow: hidden;
    margin-top: 30px;
    transition: all 0.3s ease;
  }

  .cta-button::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, #00d9ff, #ff006e);
    z-index: -1;
    transition: left 0.5s ease;
  }

  .cta-button:hover::before {
    left: 0;
  }

  .cta-button:hover {
    color: #0a0e27;
    box-shadow: 0 0 30px rgba(0, 217, 255, 0.6), 0 0 60px rgba(255, 0, 110, 0.3);
    transform: translateY(-3px);
  }

  /* COURSE MODULES GRID */
  .modules-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 25px;
    margin: 60px 0;
  }

  .module-card {
    position: relative;
    padding: 30px;
    border-radius: 15px;
    background: linear-gradient(135deg, rgba(0, 217, 255, 0.1), rgba(255, 0, 110, 0.05));
    border: 2px solid rgba(0, 217, 255, 0.3);
    cursor: pointer;
    overflow: hidden;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    animation: fadeInUp 0.8s ease-out both;
  }

  .module-card:nth-child(1) { animation-delay: 0.1s; }
  .module-card:nth-child(2) { animation-delay: 0.2s; }
  .module-card:nth-child(3) { animation-delay: 0.3s; }
  .module-card:nth-child(4) { animation-delay: 0.4s; }
  .module-card:nth-child(5) { animation-delay: 0.5s; }
  .module-card:nth-child(6) { animation-delay: 0.6s; }

  @keyframes fadeInUp {
    from {
      opacity: 0;
      transform: translateY(50px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }

  .module-card::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: linear-gradient(45deg, transparent, rgba(0, 217, 255, 0.3), transparent);
    animation: shimmer 3s infinite;
  }

  @keyframes shimmer {
    0% { transform: translate(-100%, -100%) rotate(45deg); }
    100% { transform: translate(100%, 100%) rotate(45deg); }
  }

  .module-card:hover {
    transform: translateY(-15px) scale(1.05);
    border-color: #ff006e;
    background: linear-gradient(135deg, rgba(0, 217, 255, 0.2), rgba(255, 0, 110, 0.15));
    box-shadow: 0 20px 60px rgba(0, 217, 255, 0.3), 0 0 40px rgba(255, 0, 110, 0.2);
  }

  .module-icon {
    font-size: 48px;
    margin-bottom: 15px;
    animation: bounce 2s ease-in-out infinite;
  }

  .module-card:hover .module-icon {
    animation: bounce 0.6s ease-in-out infinite;
  }

  @keyframes bounce {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-10px); }
  }

  .module-title {
    font-size: 24px;
    font-weight: 700;
    margin-bottom: 12px;
    color: #00d9ff;
    position: relative;
    z-index: 2;
  }

  .module-desc {
    font-size: 14px;
    color: #b0c4de;
    line-height: 1.6;
    position: relative;
    z-index: 2;
  }

  /* STATS SECTION WITH COUNTERS */
  .stats-section {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 20px;
    margin: 80px 0;
  }

  .stat-box {
    padding: 30px;
    text-align: center;
    border-radius: 15px;
    background: linear-gradient(135deg, rgba(255, 190, 11, 0.1), rgba(0, 217, 255, 0.1));
    border: 2px solid rgba(255, 190, 11, 0.3);
    animation: pulse 2s ease-in-out infinite;
  }

  .stat-box:nth-child(odd) {
    border-color: rgba(0, 217, 255, 0.3);
  }

  .stat-box:nth-child(even) {
    border-color: rgba(255, 0, 110, 0.3);
  }

  @keyframes pulse {
    0%, 100% { box-shadow: 0 0 20px rgba(0, 217, 255, 0.2); }
    50% { box-shadow: 0 0 40px rgba(0, 217, 255, 0.4); }
  }

  .stat-number {
    font-size: 48px;
    font-weight: 900;
    background: linear-gradient(45deg, #00d9ff, #ffbe0b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 10px;
  }

  .stat-label {
    font-size: 14px;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #b0c4de;
    font-weight: 600;
  }

  /* ALGORITHM SHOWCASE WITH MORPHING ANIMATION */
  .algorithm-showcase {
    margin: 80px 0;
    padding: 50px;
    border-radius: 20px;
    background: linear-gradient(135deg, rgba(0, 217, 255, 0.15), rgba(255, 0, 110, 0.1));
    border: 2px solid rgba(0, 217, 255, 0.4);
    position: relative;
    overflow: hidden;
  }

  .algorithm-showcase::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 200%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(0, 217, 255, 0.2), transparent);
    animation: sweep 4s infinite;
  }

  @keyframes sweep {
    0% { left: -100%; }
    50% { left: 100%; }
    100% { left: 100%; }
  }

  .showcase-title {
    font-size: 36px;
    font-weight: 800;
    color: #00d9ff;
    margin-bottom: 30px;
    position: relative;
    z-index: 2;
    text-transform: uppercase;
    letter-spacing: 3px;
  }

  .algorithm-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 15px;
    position: relative;
    z-index: 2;
  }

  .algorithm-chip {
    padding: 15px 20px;
    border-radius: 50px;
    background: rgba(0, 217, 255, 0.1);
    border: 2px solid rgba(0, 217, 255, 0.5);
    color: #00d9ff;
    font-weight: 600;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
  }

  .algorithm-chip::after {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 0;
    height: 0;
    border-radius: 50%;
    background: rgba(0, 217, 255, 0.3);
    transform: translate(-50%, -50%);
    transition: width 0.6s, height 0.6s;
  }

  .algorithm-chip:hover::after {
    width: 300px;
    height: 300px;
  }

  .algorithm-chip:hover {
    transform: scale(1.1);
    border-color: #ffbe0b;
    color: #ffbe0b;
    box-shadow: 0 0 30px rgba(0, 217, 255, 0.5);
  }

  /* SECTION HEADERS */
  .section-header {
    font-size: 44px;
    font-weight: 900;
    margin: 80px 0 40px;
    text-transform: uppercase;
    letter-spacing: 3px;
    position: relative;
    padding-bottom: 20px;
  }

  .section-header::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    width: 0;
    height: 4px;
    background: linear-gradient(90deg, #00d9ff, #ffbe0b, #ff006e);
    animation: expandWidth 0.8s ease-out forwards;
  }

  @keyframes expandWidth {
    from { width: 0; }
    to { width: 100px; }
  }

  .section-header span {
    background: linear-gradient(45deg, #00d9ff, #ffbe0b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }

  /* FLOATING ELEMENTS */
  .floating-badge {
    display: inline-block;
    padding: 10px 20px;
    border-radius: 30px;
    background: rgba(0, 217, 255, 0.15);
    border: 1px solid #00d9ff;
    color: #00d9ff;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-right: 10px;
    margin-bottom: 20px;
    animation: floatUp 3s ease-in-out infinite;
  }

  .floating-badge:nth-child(2) { animation-delay: 0.3s; }
  .floating-badge:nth-child(3) { animation-delay: 0.6s; }

  @keyframes floatUp {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-20px); }
  }

  /* FOOTER STYLING */
  .footer {
    text-align: center;
    margin-top: 100px;
    padding: 50px 0;
    border-top: 2px solid rgba(0, 217, 255, 0.2);
    animation: fadeInUp 1s ease-out 1s both;
  }

  .footer-text {
    font-size: 16px;
    color: #b0c4de;
    margin-bottom: 20px;
    letter-spacing: 1px;
  }

  .footer-highlight {
    color: #00d9ff;
    font-weight: 700;
  }

  /* RESPONSIVE */
  @media (max-width: 768px) {
    .glowing-title {
      font-size: 48px;
    }

    .subtitle {
      font-size: 20px;
    }

    .modules-grid {
      grid-template-columns: 1fr;
    }

    .stats-section {
      grid-template-columns: repeat(2, 1fr);
    }
  }
</style>

<div class="hero">
  <div class="animated-bg shape-1"></div>
  <div class="animated-bg shape-2"></div>
  <div class="animated-bg shape-3"></div>

  <div class="title-wrapper">
    <h1 class="glowing-title">ML CLASSIFICATION</h1>
    <h2 class="subtitle">Master Machine Learning</h2>
    <div class="typing-effect">→ From Basics to Production Ready ←</div>
  </div>

  <button class="cta-button">Start Learning</button>
</div>

---

## <span>📚 Course Modules</span>

<div class="modules-grid">
  <div class="module-card">
    <div class="module-icon">🎯</div>
    <div class="module-title">Fundamentals</div>
    <div class="module-desc">Master classification basics, terminology, and real-world applications</div>
  </div>

  <div class="module-card">
    <div class="module-icon">🤖</div>
    <div class="module-title">Algorithms</div>
    <div class="module-desc">Learn 15+ classification algorithms with hands-on implementation</div>
  </div>

  <div class="module-card">
    <div class="module-icon">📊</div>
    <div class="module-title">Evaluation</div>
    <div class="module-desc">Master metrics, validation, and optimization techniques</div>
  </div>

  <div class="module-card">
    <div class="module-icon">🔨</div>
    <div class="module-title">Feature Engineering</div>
    <div class="module-desc">Transform raw data into powerful predictive features</div>
  </div>

  <div class="module-card">
    <div class="module-icon">💻</div>
    <div class="module-title">Projects</div>
    <div class="module-desc">Build 10+ real-world projects from scratch</div>
  </div>

  <div class="module-card">
    <div class="module-icon">🚀</div>
    <div class="module-title">Deployment</div>
    <div class="module-desc">Deploy models to production and scale them</div>
  </div>
</div>

---

## <span>⚡ Key Statistics</span>

<div class="stats-section">
  <div class="stat-box">
    <div class="stat-number">50+</div>
    <div class="stat-label">Video Tutorials</div>
  </div>

  <div class="stat-box">
    <div class="stat-number">30+</div>
    <div class="stat-label">Code Examples</div>
  </div>

  <div class="stat-box">
    <div class="stat-number">15+</div>
    <div class="stat-label">Datasets</div>
  </div>

  <div class="stat-box">
    <div class="stat-number">100%</div>
    <div class="stat-label">Coverage</div>
  </div>
</div>

---

## <span>🧠 Classification Algorithms</span>

<div class="algorithm-showcase">
  <h3 class="showcase-title">⚙️ Available Algorithms</h3>
  <div class="algorithm-grid">
    <div class="algorithm-chip">Logistic Regression</div>
    <div class="algorithm-chip">Decision Trees</div>
    <div class="algorithm-chip">Random Forest</div>
    <div class="algorithm-chip">K-Nearest Neighbors</div>
    <div class="algorithm-chip">Naive Bayes</div>
    <div class="algorithm-chip">Support Vector Machine</div>
    <div class="algorithm-chip">Gradient Boosting</div>
    <div class="algorithm-chip">Neural Networks</div>
    <div class="algorithm-chip">XGBoost</div>
  </div>
</div>

---

## <span>🎖️ Achievements</span>

<div class="floating-badge">✓ Industry Standard</div>
<div class="floating-badge">✓ Expert Crafted</div>
<div class="floating-badge">✓ Production Ready</div>

By completing this course, you will:
- ✨ Understand advanced classification concepts
- ✨ Implement production-grade models
- ✨ Master data preprocessing pipelines
- ✨ Optimize hyperparameters like a pro
- ✨ Deploy models to production
- ✨ Handle real-world challenges

---

<div class="footer">
  <p class="footer-text">
    🚀 <span class="footer-highlight">ML Classification Masterclass</span> 🚀
  </p>
  <p class="footer-text">
    Made with ❤️ for Data Science Enthusiasts | By WsCube Tech
  </p>
  <p class="footer-text">
    <span class="footer-highlight">Master the Art of Prediction</span> ✨
  </p>
</div>
