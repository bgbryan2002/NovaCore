import markdown

with open('meeting_report.md', 'r', encoding='utf-8') as f:
    md_content = f.read()

html_content = markdown.markdown(md_content)

with open('meeting_report.html', 'w', encoding='utf-8') as f:
    f.write(f'''
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Meeting Summary Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        h3 {{ color: #7f8c8d; }}
        ul {{ list-style-type: disc; }}
    </style>
</head>
<body>
{html_content}
</body>
</html>
''') 