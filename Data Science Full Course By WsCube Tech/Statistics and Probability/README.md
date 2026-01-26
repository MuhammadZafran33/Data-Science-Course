## this  Section include Probability and statistics Concepts explain with physical implemetation of code and Graphs.

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Statistics & Probability Course</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            line-height: 1.6;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            background: white;
            border-radius: 15px;
            padding: 40px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            text-align: center;
        }
        
        header h1 {
            color: #667eea;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        header p {
            color: #666;
            font-size: 1.1em;
        }
        
        .course-status {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px 30px;
            border-radius: 10px;
            margin-top: 20px;
            font-weight: 500;
            display: inline-block;
        }
        
        .progress-container {
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        
        .progress-title {
            color: #667eea;
            font-size: 1.5em;
            margin-bottom: 20px;
            font-weight: 600;
        }
        
        .chart-wrapper {
            position: relative;
            height: 300px;
            margin-bottom: 20px;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        
        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        
        .stat-number {
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .stat-label {
            font-size: 0.9em;
            opacity: 0.9;
        }
        
        .content-section {
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        
        .section-title {
            color: #667eea;
            font-size: 1.8em;
            margin-bottom: 25px;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        
        thead {
            background: #667eea;
            color: white;
        }
        
        th, td {
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        
        tr:hover {
            background: #f5f5f5;
        }
        
        .status-badge {
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
        }
        
        .status-completed {
            background: #4CAF50;
            color: white;
        }
        
        .status-inprogress {
            background: #FFC107;
            color: black;
        }
        
        .status-pending {
            background: #f44336;
            color: white;
        }
        
        .feature-list {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .feature-card {
            background: #f9f9f9;
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }
        
        .feature-card h3 {
            color: #667eea;
            margin-bottom: 10px;
        }
        
        .feature-card p {
            color: #666;
            font-size: 0.95em;
        }
        
        .icon {
            font-size: 2em;
            margin-bottom: 10px;
        }
        
        footer {
            text-align: center;
            padding: 20px;
            color: white;
            margin-top: 30px;
        }
        
        .highlight-box {
            background: #f0f4ff;
            border-left: 4px solid #667eea;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 Statistics & Probability</h1>
            <p>A comprehensive journey through statistical concepts and probability theory</p>
            <div class="course-status">
                🔄 Course in Progress - INSHALLAH Coming Soon!
            </div>
        </header>
        
        <div class="progress-container">
            <div class="progress-title">📈 Course Progress Overview</div>
            <div class="chart-wrapper">
                <canvas id="progressChart"></canvas>
            </div>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-number">12</div>
                    <div class="stat-label">Total Modules</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">8</div>
                    <div class="stat-label">Completed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">4</div>
                    <div class="stat-label">In Progress</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">67%</div>
                    <div class="stat-label">Overall Progress</div>
                </div>
            </div>
        </div>
        
        <div class="content-section">
            <div class="section-title">📚 Course Curriculum</div>
            <table>
                <thead>
                    <tr>
                        <th>Module</th>
                        <th>Topic</th>
                        <th>Status</th>
                        <th>Progress</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>01</td>
                        <td>Descriptive Statistics & Data Analysis</td>
                        <td><span class="status-badge status-completed">✓ Completed</span></td>
                        <td>100%</td>
                    </tr>
                    <tr>
                        <td>02</td>
                        <td>Measures of Central Tendency</td>
                        <td><span class="status-badge status-completed">✓ Completed</span></td>
                        <td>100%</td>
                    </tr>
                    <tr>
                        <td>03</td>
                        <td>Measures of Dispersion</td>
                        <td><span class="status-badge status-completed">✓ Completed</span></td>
                        <td>100%</td>
                    </tr>
                    <tr>
                        <td>04</td>
                        <td>Probability Fundamentals</td>
                        <td><span class="status-badge status-completed">✓ Completed</span></td>
                        <td>100%</td>
                    </tr>
                    <tr>
                        <td>05</td>
                        <td>Probability Distributions</td>
                        <td><span class="status-badge status-inprogress">🔄 In Progress</span></td>
                        <td>75%</td>
                    </tr>
                    <tr>
                        <td>06</td>
                        <td>Normal Distribution & Z-scores</td>
                        <td><span class="status-badge status-inprogress">🔄 In Progress</span></td>
                        <td>60%</td>
                    </tr>
                    <tr>
                        <td>07</td>
                        <td>Hypothesis Testing</td>
                        <td><span class="status-badge status-completed">✓ Completed</span></td>
                        <td>100%</td>
                    </tr>
                    <tr>
                        <td>08</td>
                        <td>Confidence Intervals</td>
                        <td><span class="status-badge status-completed">✓ Completed</span></td>
                        <td>100%</td>
                    </tr>
                    <tr>
                        <td>09</td>
                        <td>Correlation & Regression</td>
                        <td><span class="status-badge status-inprogress">🔄 In Progress</span></td>
                        <td>50%</td>
                    </tr>
                    <tr>
                        <td>10</td>
                        <td>Chi-Square & ANOVA Tests</td>
                        <td><span class="status-badge status-inprogress">🔄 In Progress</span></td>
                        <td>40%</td>
                    </tr>
                    <tr>
                        <td>11</td>
                        <td>Bayesian Statistics</td>
                        <td><span class="status-badge status-pending">⏳ Coming Soon</span></td>
                        <td>0%</td>
                    </tr>
                    <tr>
                        <td>12</td>
                        <td>Time Series Analysis</td>
                        <td><span class="status-badge status-pending">⏳ Coming Soon</span></td>
                        <td>0%</td>
                    </tr>
                </tbody>
            </table>
        </div>
        
        <div class="content-section">
            <div class="section-title">✨ Key Features</div>
            <div class="feature-list">
                <div class="feature-card">
                    <div class="icon">📖</div>
                    <h3>Comprehensive Theory</h3>
                    <p>Deep dive into statistical concepts with clear explanations and mathematical foundations</p>
                </div>
                <div class="feature-card">
                    <div class="icon">💻</div>
                    <h3>Hands-on Practice</h3>
                    <p>Practical examples and coding exercises with real-world datasets</p>
                </div>
                <div class="feature-card">
                    <div class="icon">📊</div>
                    <h3>Visual Learning</h3>
                    <p>Charts, graphs, and interactive visualizations for better understanding</p>
                </div>
                <div class="feature-card">
                    <div class="icon">🎯</div>
                    <h3>Problem Solving</h3>
                    <p>Step-by-step solutions to complex statistical problems</p>
                </div>
                <div class="feature-card">
                    <div class="icon">🔬</div>
                    <h3>Real-world Applications</h3>
                    <p>Learn how statistics is used in business, science, and data analysis</p>
                </div>
                <div class="feature-card">
                    <div class="icon">✅</div>
                    <h3>Assessments</h3>
                    <p>Quizzes and assignments to test your understanding</p>
                </div>
            </div>
        </div>
        
        <div class="content-section">
            <div class="section-title">📊 Learning Distribution</div>
            <div class="chart-wrapper">
                <canvas id="distributionChart"></canvas>
            </div>
        </div>
        
        <div class="content-section">
            <div class="section-title">🎓 What You'll Learn</div>
            <div class="highlight-box">
                <strong>Foundations:</strong> Master the basics of descriptive statistics, probability, and data interpretation
            </div>
            <div class="highlight-box">
                <strong>Inference:</strong> Learn hypothesis testing, confidence intervals, and statistical inference techniques
            </div>
            <div class="highlight-box">
                <strong>Relationships:</strong> Explore correlation, regression, and multivariate analysis methods
            </div>
            <div class="highlight-box">
                <strong>Advanced Topics:</strong> Dive into Bayesian methods, time series, and advanced statistical techniques
            </div>
        </div>
        
        <div class="content-section">
            <div class="section-title">📞 Contact & Resources</div>
            <p><strong>Course Creator:</strong> Muhammad Zafran</p>
            <p><strong>Repository:</strong> <a href="https://github.com/MuhammadZafran33/Data-Science-Course" target="_blank">GitHub Repository</a></p>
            <p><strong>Last Updated:</strong> January 2025</p>
            <p style="margin-top: 20px; color: #666; font-style: italic;">📝 Note: Topics are being continuously updated. Check back regularly for new materials and updates!</p>
        </div>
        
        <footer>
            <p>🌟 Keep Learning, Keep Growing! 🌟</p>
            <p>Made with ❤️ for Data Science Enthusiasts</p>
        </footer>
    </div>
    
    <script>
        // Progress Chart
        const ctx1 = document.getElementById('progressChart').getContext('2d');
        new Chart(ctx1, {
            type: 'doughnut',
            data: {
                labels: ['Completed', 'In Progress', 'Coming Soon'],
                datasets: [{
                    data: [67, 25, 8],
                    backgroundColor: ['#4CAF50', '#FFC107', '#f44336'],
                    borderColor: ['#fff', '#fff', '#fff'],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            font: { size: 13 },
                            padding: 15
                        }
                    }
                }
            }
        });
        
        // Distribution Chart
        const ctx2 = document.getElementById('distributionChart').getContext('2d');
        new Chart(ctx2, {
            type: 'bar',
            data: {
                labels: ['Descriptive\nStats', 'Probability', 'Distributions', 'Inference', 'Regression', 'Advanced\nTopics'],
                datasets: [{
                    label: 'Hours of Content',
                    data: [12, 15, 14, 16, 13, 10],
                    backgroundColor: [
                        '#667eea',
                        '#764ba2',
                        '#f093fb',
                        '#4facfe',
                        '#00f2fe',
                        '#43e97b'
                    ],
                    borderRadius: 8,
                    borderSkipped: false
                }]
            },
            options: {
                indexAxis: 'x',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        labels: { font: { size: 12 } }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: { stepSize: 5 }
                    }
                }
            }
        });
    </script>
</body>
</html>
