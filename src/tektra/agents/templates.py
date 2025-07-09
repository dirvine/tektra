"""
Agent Template Library

This module provides pre-built agent templates for common use cases.
Templates serve as starting points that users can customize for their needs.

Templates include:
- Research agents
- Monitoring agents
- Data processing agents
- Automation agents
- Integration agents
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any
import json

from .builder import AgentSpecification, AgentType, AgentCapability


@dataclass
class AgentTemplate:
    """A reusable agent template."""
    name: str
    category: str
    description: str
    icon: str  # Emoji or icon identifier
    
    # Base specification
    base_spec: AgentSpecification
    
    # Customization points
    parameters: Dict[str, Dict[str, Any]]  # parameter_name -> {type, default, description}
    
    # Example usage
    example_input: Dict[str, Any]
    example_output: Dict[str, Any]
    
    # Documentation
    detailed_description: str
    use_cases: List[str]
    limitations: List[str]
    
    # Metadata
    author: str = "Tektra Team"
    version: str = "1.0.0"
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class TemplateLibrary:
    """
    Library of pre-built agent templates.
    
    Provides templates for common agent patterns that users
    can instantiate and customize.
    """
    
    def __init__(self):
        """Initialize template library with built-in templates."""
        self.templates: Dict[str, AgentTemplate] = {}
        self._load_builtin_templates()
    
    def _load_builtin_templates(self):
        """Load all built-in templates."""
        # Research Agent Template
        self.templates['research_agent'] = self._create_research_agent_template()
        
        # Web Monitor Agent Template
        self.templates['web_monitor'] = self._create_web_monitor_template()
        
        # Data Analyzer Agent Template
        self.templates['data_analyzer'] = self._create_data_analyzer_template()
        
        # Code Review Agent Template
        self.templates['code_reviewer'] = self._create_code_review_template()
        
        # Daily Report Agent Template
        self.templates['daily_reporter'] = self._create_daily_report_template()
        
        # API Integration Agent Template
        self.templates['api_integrator'] = self._create_api_integration_template()
        
        # File Organizer Agent Template
        self.templates['file_organizer'] = self._create_file_organizer_template()
        
        # Alert Monitor Agent Template
        self.templates['alert_monitor'] = self._create_alert_monitor_template()
    
    def _create_research_agent_template(self) -> AgentTemplate:
        """Create research agent template."""
        spec = AgentSpecification(
            name="Research Assistant",
            description="Searches and summarizes information on specified topics",
            type=AgentType.CODE,
            capabilities=[
                AgentCapability.WEB_SEARCH,
                AgentCapability.DATA_ANALYSIS
            ],
            goal="Research topics and provide comprehensive summaries",
            constraints=[
                "Only use reliable sources",
                "Cite all sources",
                "Provide balanced perspectives"
            ],
            success_criteria=[
                "Found relevant information",
                "Created comprehensive summary",
                "Included citations"
            ],
            system_prompt="""You are a research assistant specialized in finding and summarizing information.

Your approach:
1. Search for information from multiple sources
2. Evaluate source reliability
3. Synthesize findings into a coherent summary
4. Include proper citations
5. Highlight key insights and conflicting viewpoints""",
            initial_code="""
async def run_agent(input_data):
    '''Research a topic and provide summary'''
    topic = input_data.get('topic', '')
    depth = input_data.get('depth', 'moderate')  # basic, moderate, comprehensive
    
    # Search for information
    search_results = await search_web(topic, num_results=10)
    
    # Analyze and summarize
    summary = await analyze_sources(search_results, depth)
    
    # Extract key insights
    insights = await extract_insights(summary)
    
    return {
        'topic': topic,
        'summary': summary,
        'insights': insights,
        'sources': [r['url'] for r in search_results],
        'timestamp': datetime.now().isoformat()
    }
""",
            trigger_type="manual",
            max_runtime_seconds=180
        )
        
        return AgentTemplate(
            name="Research Assistant",
            category="Research & Analysis",
            description="Automated research on any topic with source citations",
            icon="🔍",
            base_spec=spec,
            parameters={
                "topic": {
                    "type": "string",
                    "default": "",
                    "description": "The topic to research",
                    "required": True
                },
                "depth": {
                    "type": "enum",
                    "values": ["basic", "moderate", "comprehensive"],
                    "default": "moderate",
                    "description": "How deep to research"
                },
                "max_sources": {
                    "type": "integer",
                    "default": 10,
                    "description": "Maximum number of sources to analyze"
                }
            },
            example_input={
                "topic": "quantum computing applications",
                "depth": "moderate"
            },
            example_output={
                "summary": "Quantum computing is revolutionizing...",
                "insights": ["Key insight 1", "Key insight 2"],
                "sources": ["https://example.com/quantum"]
            },
            detailed_description="""
The Research Assistant agent automates the process of researching any topic.
It searches multiple sources, evaluates their reliability, and synthesizes
the information into a comprehensive summary with proper citations.

Perfect for:
- Market research
- Technical research
- Academic research
- Competitive analysis
- Trend analysis
""",
            use_cases=[
                "Research emerging technologies",
                "Analyze market trends",
                "Gather competitive intelligence",
                "Create literature reviews",
                "Fact-check claims"
            ],
            limitations=[
                "Limited to publicly available information",
                "May not access paywalled content",
                "Quality depends on available sources"
            ]
        )
    
    def _create_web_monitor_template(self) -> AgentTemplate:
        """Create web monitoring agent template."""
        spec = AgentSpecification(
            name="Web Monitor",
            description="Monitors websites for changes and alerts on updates",
            type=AgentType.MONITOR,
            capabilities=[
                AgentCapability.WEB_SEARCH,
                AgentCapability.NOTIFICATIONS
            ],
            goal="Monitor specified websites and alert on changes",
            constraints=[
                "Respect robots.txt",
                "Don't overload servers",
                "Only monitor public pages"
            ],
            success_criteria=[
                "Detected changes accurately",
                "Sent timely notifications",
                "Minimized false positives"
            ],
            system_prompt="You are a web monitoring agent that tracks changes on websites.",
            initial_code="""
async def run_agent(input_data):
    '''Monitor websites for changes'''
    urls = input_data.get('urls', [])
    change_threshold = input_data.get('change_threshold', 0.1)
    
    changes_detected = []
    
    for url in urls:
        # Fetch current content
        current_content = await fetch_webpage(url)
        
        # Compare with previous version
        previous_content = await get_cached_content(url)
        
        if previous_content:
            change_ratio = calculate_change_ratio(previous_content, current_content)
            
            if change_ratio > change_threshold:
                changes_detected.append({
                    'url': url,
                    'change_ratio': change_ratio,
                    'summary': summarize_changes(previous_content, current_content)
                })
        
        # Cache current content
        await cache_content(url, current_content)
    
    # Send notifications if changes detected
    if changes_detected:
        await send_notifications(changes_detected)
    
    return {
        'monitored_urls': len(urls),
        'changes_detected': len(changes_detected),
        'changes': changes_detected,
        'timestamp': datetime.now().isoformat()
    }
""",
            trigger_type="scheduled",
            schedule="*/30 * * * *",  # Every 30 minutes
            max_runtime_seconds=120
        )
        
        return AgentTemplate(
            name="Web Monitor",
            category="Monitoring & Alerts",
            description="Monitor websites for changes and get notified",
            icon="👁️",
            base_spec=spec,
            parameters={
                "urls": {
                    "type": "array",
                    "default": [],
                    "description": "List of URLs to monitor",
                    "required": True
                },
                "change_threshold": {
                    "type": "float",
                    "default": 0.1,
                    "description": "Minimum change ratio to trigger alert (0-1)"
                },
                "check_frequency": {
                    "type": "string",
                    "default": "30m",
                    "description": "How often to check (e.g., 30m, 1h, 1d)"
                }
            },
            example_input={
                "urls": ["https://example.com/pricing", "https://competitor.com/features"],
                "change_threshold": 0.15
            },
            example_output={
                "monitored_urls": 2,
                "changes_detected": 1,
                "changes": [{
                    "url": "https://example.com/pricing",
                    "change_ratio": 0.23,
                    "summary": "Price increased from $99 to $119"
                }]
            },
            detailed_description="""
The Web Monitor agent continuously tracks specified web pages for changes.
It compares page content over time and alerts you when significant changes occur.

Use it to monitor:
- Competitor pricing pages
- Product availability
- News sections
- Documentation updates
- Job postings
""",
            use_cases=[
                "Monitor competitor pricing",
                "Track product availability",
                "Watch for documentation updates",
                "Alert on news mentions",
                "Track regulatory changes"
            ],
            limitations=[
                "Cannot monitor pages requiring login",
                "May miss JavaScript-rendered content",
                "Rate limited to avoid overloading servers"
            ]
        )
    
    def _create_data_analyzer_template(self) -> AgentTemplate:
        """Create data analysis agent template."""
        spec = AgentSpecification(
            name="Data Analyzer",
            description="Analyzes data files and generates insights",
            type=AgentType.CODE,
            capabilities=[
                AgentCapability.FILE_ACCESS,
                AgentCapability.DATA_ANALYSIS,
                AgentCapability.IMAGE_PROCESSING
            ],
            goal="Analyze data and generate actionable insights",
            constraints=[
                "Handle data privacy appropriately",
                "Validate data quality",
                "Provide clear visualizations"
            ],
            success_criteria=[
                "Successfully loaded data",
                "Generated meaningful insights",
                "Created clear visualizations"
            ],
            system_prompt="You are a data analysis expert that finds insights in data.",
            initial_code="""
async def run_agent(input_data):
    '''Analyze data and generate insights'''
    file_path = input_data.get('file_path')
    analysis_type = input_data.get('analysis_type', 'auto')
    
    # Load data
    data = await load_data_file(file_path)
    
    # Perform analysis based on type
    if analysis_type == 'auto':
        analysis_type = detect_best_analysis(data)
    
    results = await perform_analysis(data, analysis_type)
    
    # Generate visualizations
    charts = await create_visualizations(data, results)
    
    # Extract key insights
    insights = await generate_insights(results)
    
    return {
        'file': file_path,
        'data_shape': data.shape if hasattr(data, 'shape') else len(data),
        'analysis_type': analysis_type,
        'insights': insights,
        'statistics': results,
        'visualizations': charts,
        'timestamp': datetime.now().isoformat()
    }
""",
            trigger_type="manual",
            max_runtime_seconds=300
        )
        
        return AgentTemplate(
            name="Data Analyzer",
            category="Data & Analytics",
            description="Analyze data files and extract insights automatically",
            icon="📊",
            base_spec=spec,
            parameters={
                "file_path": {
                    "type": "string",
                    "default": "",
                    "description": "Path to data file (CSV, JSON, Excel)",
                    "required": True
                },
                "analysis_type": {
                    "type": "enum",
                    "values": ["auto", "statistical", "trend", "correlation", "clustering"],
                    "default": "auto",
                    "description": "Type of analysis to perform"
                },
                "generate_report": {
                    "type": "boolean",
                    "default": True,
                    "description": "Generate detailed report"
                }
            },
            example_input={
                "file_path": "/data/sales_2024.csv",
                "analysis_type": "trend"
            },
            example_output={
                "data_shape": [1000, 12],
                "insights": [
                    "Sales increased 23% in Q3",
                    "Product A shows strongest growth",
                    "Weekend sales outperform weekdays"
                ],
                "visualizations": ["sales_trend.png", "product_comparison.png"]
            },
            detailed_description="""
The Data Analyzer agent automatically analyzes your data files to find patterns,
trends, and insights. It supports various file formats and can perform different
types of analysis based on your data.

Features:
- Automatic data type detection
- Statistical analysis
- Trend identification
- Correlation analysis
- Automated visualization generation
""",
            use_cases=[
                "Analyze sales data",
                "Find patterns in user behavior",
                "Identify trends in metrics",
                "Generate automated reports",
                "Discover data anomalies"
            ],
            limitations=[
                "Limited to structured data formats",
                "Large files may take longer",
                "Complex analyses may need customization"
            ]
        )
    
    def _create_code_review_template(self) -> AgentTemplate:
        """Create code review agent template."""
        spec = AgentSpecification(
            name="Code Reviewer",
            description="Reviews code for quality, security, and best practices",
            type=AgentType.CODE,
            capabilities=[
                AgentCapability.FILE_ACCESS,
                AgentCapability.CODE_EXECUTION
            ],
            goal="Review code and provide constructive feedback",
            constraints=[
                "Focus on constructive feedback",
                "Check for security vulnerabilities",
                "Suggest improvements, not just problems"
            ],
            success_criteria=[
                "Identified potential issues",
                "Provided actionable suggestions",
                "Checked security best practices"
            ],
            system_prompt="""You are an expert code reviewer focused on improving code quality.

Review for:
1. Code style and readability
2. Potential bugs and edge cases
3. Security vulnerabilities
4. Performance issues
5. Best practices and patterns""",
            initial_code="""
async def run_agent(input_data):
    '''Review code files for quality and security'''
    file_paths = input_data.get('file_paths', [])
    review_level = input_data.get('review_level', 'standard')
    
    all_issues = []
    
    for file_path in file_paths:
        # Read code file
        code_content = await read_file(file_path)
        
        # Analyze code
        issues = await analyze_code(code_content, review_level)
        
        # Check security
        security_issues = await check_security(code_content)
        issues.extend(security_issues)
        
        # Suggest improvements
        suggestions = await generate_suggestions(code_content, issues)
        
        all_issues.append({
            'file': file_path,
            'issues': issues,
            'suggestions': suggestions,
            'severity_counts': count_by_severity(issues)
        })
    
    # Generate summary report
    summary = await generate_review_summary(all_issues)
    
    return {
        'files_reviewed': len(file_paths),
        'total_issues': sum(len(f['issues']) for f in all_issues),
        'file_results': all_issues,
        'summary': summary,
        'timestamp': datetime.now().isoformat()
    }
""",
            trigger_type="manual",
            max_runtime_seconds=180
        )
        
        return AgentTemplate(
            name="Code Reviewer",
            category="Development & DevOps",
            description="Automated code review for quality and security",
            icon="🔍",
            base_spec=spec,
            parameters={
                "file_paths": {
                    "type": "array",
                    "default": [],
                    "description": "List of code files to review",
                    "required": True
                },
                "review_level": {
                    "type": "enum",
                    "values": ["basic", "standard", "thorough"],
                    "default": "standard",
                    "description": "How thorough the review should be"
                },
                "languages": {
                    "type": "array",
                    "default": ["auto"],
                    "description": "Programming languages to check"
                }
            },
            example_input={
                "file_paths": ["src/auth.py", "src/api.py"],
                "review_level": "thorough"
            },
            example_output={
                "files_reviewed": 2,
                "total_issues": 7,
                "summary": "Found 2 security issues and 5 code quality improvements"
            },
            detailed_description="""
The Code Reviewer agent automatically reviews your code for quality,
security vulnerabilities, and adherence to best practices.

It checks for:
- Security vulnerabilities (SQL injection, XSS, etc.)
- Code style and readability
- Potential bugs and edge cases
- Performance issues
- Best practices for the language
""",
            use_cases=[
                "Pre-commit code review",
                "Security vulnerability scanning",
                "Code quality assessment",
                "Best practices enforcement",
                "Technical debt identification"
            ],
            limitations=[
                "Cannot execute code in production",
                "May miss context-specific issues",
                "Language support varies"
            ]
        )
    
    def _create_daily_report_template(self) -> AgentTemplate:
        """Create daily report agent template."""
        spec = AgentSpecification(
            name="Daily Reporter",
            description="Generates daily reports from various data sources",
            type=AgentType.WORKFLOW,
            capabilities=[
                AgentCapability.DATABASE,
                AgentCapability.EMAIL,
                AgentCapability.DATA_ANALYSIS
            ],
            goal="Compile and send comprehensive daily reports",
            constraints=[
                "Ensure data accuracy",
                "Maintain consistent format",
                "Send at scheduled time"
            ],
            success_criteria=[
                "Collected all required data",
                "Generated formatted report",
                "Sent to all recipients"
            ],
            system_prompt="You create comprehensive daily reports from multiple data sources.",
            initial_code="""
async def run_agent(input_data):
    '''Generate and send daily report'''
    report_config = input_data.get('config', {})
    
    # Collect data from sources
    metrics = await collect_metrics(report_config['data_sources'])
    
    # Analyze trends
    trends = await analyze_daily_trends(metrics)
    
    # Generate report sections
    sections = {
        'summary': await generate_summary(metrics),
        'key_metrics': await format_key_metrics(metrics),
        'trends': await format_trends(trends),
        'alerts': await check_alerts(metrics, report_config['alert_rules'])
    }
    
    # Format report
    report = await format_report(sections, report_config['template'])
    
    # Send report
    recipients = report_config.get('recipients', [])
    await send_email_report(report, recipients)
    
    return {
        'report_generated': True,
        'metrics_collected': len(metrics),
        'recipients': len(recipients),
        'alerts_triggered': len(sections['alerts']),
        'timestamp': datetime.now().isoformat()
    }
""",
            trigger_type="scheduled",
            schedule="0 9 * * *",  # 9 AM daily
            max_runtime_seconds=300
        )
        
        return AgentTemplate(
            name="Daily Reporter",
            category="Reporting & Analytics",
            description="Automated daily reports with metrics and insights",
            icon="📈",
            base_spec=spec,
            parameters={
                "data_sources": {
                    "type": "array",
                    "default": [],
                    "description": "Data sources to include",
                    "required": True
                },
                "recipients": {
                    "type": "array",
                    "default": [],
                    "description": "Email addresses for report delivery",
                    "required": True
                },
                "report_time": {
                    "type": "string",
                    "default": "09:00",
                    "description": "Time to send daily report"
                }
            },
            example_input={
                "config": {
                    "data_sources": ["sales_db", "analytics_api"],
                    "recipients": ["team@company.com"],
                    "template": "executive_summary"
                }
            },
            example_output={
                "report_generated": True,
                "metrics_collected": 25,
                "recipients": 3,
                "alerts_triggered": 2
            },
            detailed_description="""
The Daily Reporter agent automatically compiles and sends comprehensive
reports based on your data sources. It can aggregate metrics, identify
trends, and highlight important changes.

Perfect for:
- Executive dashboards
- Team performance reports
- Sales summaries
- System health reports
- KPI tracking
""",
            use_cases=[
                "Daily business metrics",
                "Team performance summaries",
                "System health reports",
                "Sales pipeline updates",
                "Customer activity reports"
            ],
            limitations=[
                "Requires data source configuration",
                "Email server setup needed",
                "Time zone considerations"
            ]
        )
    
    def _create_api_integration_template(self) -> AgentTemplate:
        """Create API integration agent template."""
        spec = AgentSpecification(
            name="API Integrator",
            description="Integrates and synchronizes data between APIs",
            type=AgentType.WORKFLOW,
            capabilities=[
                AgentCapability.API_CALLS,
                AgentCapability.DATA_ANALYSIS
            ],
            goal="Synchronize data between different systems via APIs",
            constraints=[
                "Respect API rate limits",
                "Handle errors gracefully",
                "Maintain data integrity"
            ],
            success_criteria=[
                "Successfully connected to APIs",
                "Data synchronized accurately",
                "No data loss or corruption"
            ],
            system_prompt="You integrate different systems by synchronizing data through their APIs.",
            initial_code="""
async def run_agent(input_data):
    '''Sync data between APIs'''
    source_api = input_data.get('source_api')
    target_api = input_data.get('target_api')
    sync_config = input_data.get('sync_config', {})
    
    # Fetch data from source
    source_data = await fetch_from_api(
        source_api['url'],
        source_api['auth'],
        sync_config.get('query', {})
    )
    
    # Transform data
    transformed_data = await transform_data(
        source_data,
        sync_config.get('mapping', {})
    )
    
    # Push to target
    results = await push_to_api(
        target_api['url'],
        target_api['auth'],
        transformed_data
    )
    
    # Verify sync
    verification = await verify_sync(source_data, results)
    
    return {
        'records_synced': len(transformed_data),
        'success_rate': verification['success_rate'],
        'errors': verification.get('errors', []),
        'timestamp': datetime.now().isoformat()
    }
""",
            trigger_type="scheduled",
            schedule="*/15 * * * *",  # Every 15 minutes
            max_runtime_seconds=180
        )
        
        return AgentTemplate(
            name="API Integrator",
            category="Integration & Automation",
            description="Sync data between different APIs automatically",
            icon="🔄",
            base_spec=spec,
            parameters={
                "source_api": {
                    "type": "object",
                    "default": {},
                    "description": "Source API configuration",
                    "required": True
                },
                "target_api": {
                    "type": "object",
                    "default": {},
                    "description": "Target API configuration",
                    "required": True
                },
                "sync_frequency": {
                    "type": "string",
                    "default": "15m",
                    "description": "How often to sync"
                }
            },
            example_input={
                "source_api": {
                    "url": "https://api.source.com/data",
                    "auth": {"type": "bearer", "token": "xxx"}
                },
                "target_api": {
                    "url": "https://api.target.com/import",
                    "auth": {"type": "api_key", "key": "yyy"}
                }
            },
            example_output={
                "records_synced": 150,
                "success_rate": 0.98,
                "errors": ["2 records failed validation"]
            },
            detailed_description="""
The API Integrator agent automatically synchronizes data between different
systems using their APIs. It handles authentication, data transformation,
and error recovery.

Use it for:
- CRM to marketing tool sync
- Database to analytics platform
- E-commerce to inventory system
- Payment processor reconciliation
""",
            use_cases=[
                "Sync CRM contacts to email platform",
                "Update inventory across systems",
                "Synchronize customer data",
                "Aggregate data from multiple sources",
                "Automate data pipelines"
            ],
            limitations=[
                "Requires API credentials",
                "Subject to API rate limits",
                "Complex transformations may need customization"
            ]
        )
    
    def _create_file_organizer_template(self) -> AgentTemplate:
        """Create file organization agent template."""
        spec = AgentSpecification(
            name="File Organizer",
            description="Organizes files based on rules and patterns",
            type=AgentType.CODE,
            capabilities=[
                AgentCapability.FILE_ACCESS
            ],
            goal="Organize files into a clean directory structure",
            constraints=[
                "Never delete files without confirmation",
                "Maintain file integrity",
                "Create backups when needed"
            ],
            success_criteria=[
                "Files organized correctly",
                "No data loss",
                "Clear organization structure"
            ],
            system_prompt="You organize files efficiently based on type, date, and content.",
            initial_code="""
async def run_agent(input_data):
    '''Organize files in directory'''
    source_dir = input_data.get('source_directory')
    rules = input_data.get('organization_rules', 'auto')
    
    # Scan directory
    files = await scan_directory(source_dir)
    
    # Determine organization strategy
    if rules == 'auto':
        rules = await detect_best_organization(files)
    
    # Create organization plan
    plan = await create_organization_plan(files, rules)
    
    # Execute organization
    results = []
    for action in plan['actions']:
        result = await execute_file_action(action)
        results.append(result)
    
    # Generate summary
    summary = await generate_organization_summary(results)
    
    return {
        'files_processed': len(files),
        'files_moved': len([r for r in results if r['moved']]),
        'new_directories': plan['new_directories'],
        'summary': summary,
        'timestamp': datetime.now().isoformat()
    }
""",
            trigger_type="manual",
            max_runtime_seconds=300
        )
        
        return AgentTemplate(
            name="File Organizer",
            category="File Management",
            description="Automatically organize files into clean directory structures",
            icon="📁",
            base_spec=spec,
            parameters={
                "source_directory": {
                    "type": "string",
                    "default": "",
                    "description": "Directory to organize",
                    "required": True
                },
                "organization_rules": {
                    "type": "enum",
                    "values": ["auto", "by_type", "by_date", "by_project", "custom"],
                    "default": "auto",
                    "description": "How to organize files"
                },
                "create_backup": {
                    "type": "boolean",
                    "default": True,
                    "description": "Create backup before organizing"
                }
            },
            example_input={
                "source_directory": "/Users/me/Downloads",
                "organization_rules": "by_type"
            },
            example_output={
                "files_processed": 234,
                "files_moved": 220,
                "new_directories": ["Documents", "Images", "Videos", "Archives"],
                "summary": "Organized 220 files into 4 categories"
            },
            detailed_description="""
The File Organizer agent automatically organizes messy directories into
clean, logical structures. It can sort by file type, date, project, or
custom rules you define.

Features:
- Smart file type detection
- Date-based organization
- Duplicate detection
- Safe operation with backups
- Custom organization rules
""",
            use_cases=[
                "Clean up Downloads folder",
                "Organize photo libraries",
                "Sort project files",
                "Archive old documents",
                "Maintain clean desktop"
            ],
            limitations=[
                "Large files may take time",
                "Cannot access system files",
                "Requires file system permissions"
            ]
        )
    
    def _create_alert_monitor_template(self) -> AgentTemplate:
        """Create alert monitoring agent template."""
        spec = AgentSpecification(
            name="Alert Monitor",
            description="Monitors metrics and sends alerts on anomalies",
            type=AgentType.MONITOR,
            capabilities=[
                AgentCapability.DATABASE,
                AgentCapability.NOTIFICATIONS,
                AgentCapability.DATA_ANALYSIS
            ],
            goal="Monitor metrics and alert on anomalies or thresholds",
            constraints=[
                "Avoid alert fatigue",
                "Validate anomalies before alerting",
                "Provide context in alerts"
            ],
            success_criteria=[
                "Detected real anomalies",
                "Sent timely alerts",
                "Minimal false positives"
            ],
            system_prompt="You monitor metrics and intelligently alert on important changes.",
            initial_code="""
async def run_agent(input_data):
    '''Monitor metrics and send alerts'''
    metrics_config = input_data.get('metrics', [])
    alert_rules = input_data.get('alert_rules', {})
    
    alerts_triggered = []
    
    for metric in metrics_config:
        # Fetch current value
        current_value = await fetch_metric(metric['source'], metric['name'])
        
        # Check against thresholds
        threshold_alerts = await check_thresholds(
            metric['name'],
            current_value,
            alert_rules.get('thresholds', {})
        )
        alerts_triggered.extend(threshold_alerts)
        
        # Check for anomalies
        historical_data = await get_historical_data(metric['source'], metric['name'])
        anomalies = await detect_anomalies(current_value, historical_data)
        
        if anomalies:
            alerts_triggered.extend([{
                'type': 'anomaly',
                'metric': metric['name'],
                'value': current_value,
                'description': anomaly['description']
            } for anomaly in anomalies])
    
    # Send alerts
    if alerts_triggered:
        await send_alerts(alerts_triggered, alert_rules.get('channels', []))
    
    return {
        'metrics_monitored': len(metrics_config),
        'alerts_triggered': len(alerts_triggered),
        'alerts': alerts_triggered,
        'timestamp': datetime.now().isoformat()
    }
""",
            trigger_type="scheduled",
            schedule="*/5 * * * *",  # Every 5 minutes
            max_runtime_seconds=60
        )
        
        return AgentTemplate(
            name="Alert Monitor",
            category="Monitoring & Alerts",
            description="Monitor metrics and alert on anomalies or thresholds",
            icon="🚨",
            base_spec=spec,
            parameters={
                "metrics": {
                    "type": "array",
                    "default": [],
                    "description": "Metrics to monitor",
                    "required": True
                },
                "alert_rules": {
                    "type": "object",
                    "default": {},
                    "description": "Rules for triggering alerts",
                    "required": True
                },
                "check_frequency": {
                    "type": "string",
                    "default": "5m",
                    "description": "How often to check metrics"
                }
            },
            example_input={
                "metrics": [
                    {"source": "app_db", "name": "response_time"},
                    {"source": "app_db", "name": "error_rate"}
                ],
                "alert_rules": {
                    "thresholds": {
                        "response_time": {"max": 1000},
                        "error_rate": {"max": 0.05}
                    }
                }
            },
            example_output={
                "metrics_monitored": 2,
                "alerts_triggered": 1,
                "alerts": [{
                    "type": "threshold",
                    "metric": "response_time",
                    "value": 1250,
                    "description": "Response time exceeded 1000ms threshold"
                }]
            },
            detailed_description="""
The Alert Monitor agent continuously watches your metrics and alerts you
when something goes wrong. It can detect both threshold violations and
statistical anomalies.

Features:
- Threshold-based alerts
- Anomaly detection
- Smart alert grouping
- Multiple notification channels
- Contextual alert information
""",
            use_cases=[
                "Monitor application performance",
                "Track business metrics",
                "Watch system resources",
                "Detect security anomalies",
                "Alert on data quality issues"
            ],
            limitations=[
                "Requires metric data access",
                "Alert fatigue if misconfigured",
                "Anomaly detection needs historical data"
            ]
        )
    
    def get_template(self, template_name: str) -> Optional[AgentTemplate]:
        """Get a template by name."""
        return self.templates.get(template_name)
    
    def list_templates(self, category: Optional[str] = None) -> List[AgentTemplate]:
        """List all templates, optionally filtered by category."""
        templates = list(self.templates.values())
        
        if category:
            templates = [t for t in templates if t.category == category]
        
        return templates
    
    def get_categories(self) -> List[str]:
        """Get all available template categories."""
        categories = set()
        for template in self.templates.values():
            categories.add(template.category)
        return sorted(list(categories))
    
    def search_templates(self, query: str) -> List[AgentTemplate]:
        """Search templates by name or description."""
        query_lower = query.lower()
        results = []
        
        for template in self.templates.values():
            if (query_lower in template.name.lower() or
                query_lower in template.description.lower() or
                query_lower in template.detailed_description.lower()):
                results.append(template)
        
        return results
    
    def create_agent_from_template(
        self,
        template_name: str,
        customizations: Dict[str, Any]
    ) -> AgentSpecification:
        """
        Create an agent specification from a template with customizations.
        
        Args:
            template_name: Name of the template to use
            customizations: Parameter values to customize the template
            
        Returns:
            Customized agent specification
        """
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")
        
        # Clone the base specification
        spec = AgentSpecification(**template.base_spec.__dict__.copy())
        
        # Apply customizations
        for param_name, param_value in customizations.items():
            if param_name in template.parameters:
                # Validate parameter type
                param_info = template.parameters[param_name]
                # Apply customization logic based on parameter
                # This is simplified - real implementation would be more sophisticated
                if param_name in spec.__dict__:
                    setattr(spec, param_name, param_value)
        
        # Generate unique ID for new instance
        spec.id = str(uuid.uuid4())
        spec.created_at = datetime.now()
        
        return spec