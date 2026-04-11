import sys
with open('d:/Projects/Personal_Projects/Rag-App/rag-app/app/web_search_tool/web_search_tool.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()
with open('d:/Projects/Personal_Projects/Rag-App/rag-app/app/web_search_tool/web_search_tool.py', 'w', encoding='utf-8') as f:
    for line in lines:
        if not line.lstrip().startswith('#'):
            f.write(line)
