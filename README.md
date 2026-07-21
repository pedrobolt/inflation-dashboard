# Inflation Dashboard

Dashboard minimalista de inflação, com dados do IPCA (Brasil) e estrutura pronta para expansão para outros países (CPI dos EUA etc.).

Dados atualizados automaticamente via GitHub Actions e publicados no GitHub Pages.

## Funcionalidades

- **Resumo:** variação mensal (MoM) e anual (YoY) do IPCA e dos 9 grupos principais.
- **Destaques:** subitens com maior contribuição positiva e negativa em pontos-base (bps).
- **Dinâmico:** seletor de mês/ano para navegar no histórico.
- **Gráfico:** evolução histórica do IPCA mensal e acumulado em 12 meses.

## Tecnologias

- Python (ETL)
- IBGE SIDRA (dados do IPCA)
- GitHub Actions (atualização automática)
- GitHub Pages (hospedagem)
- Plotly.js (gráficos interativos)

## Estrutura

```
inflation-dashboard/
├── src/
│   ├── collectors/brazil/   # Coleta de dados do IBGE
│   ├── processors/brazil/   # Processamento de Resumo e Destaques
│   ├── builders/            # Geração do site estático
│   └── builders/templates/  # Template HTML
├── site/                    # Site gerado (publicado no GitHub Pages)
│   ├── index.html
│   ├── css/
│   ├── js/
│   └── data/brazil/
├── .github/workflows/       # Automação
├── tests/
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Como rodar localmente

1. Clone o repositório:

```bash
git clone https://github.com/seu-usuario/inflation-dashboard.git
cd inflation-dashboard
```

2. Crie um ambiente virtual (opcional, mas recomendado):

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

3. Instale as dependências:

```bash
pip install -r requirements.txt
```

4. Execute o pipeline:

```bash
python -m src.builders.generate_site
```

5. Sirva o site localmente:

```bash
cd site
python -m http.server 8000
```

6. Abra http://localhost:8000 no navegador.

## Como publicar no GitHub Pages

1. Crie um repositório público no GitHub.
2. Envie o código:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/seu-usuario/inflation-dashboard.git
git push -u origin main
```

3. No GitHub, vá em **Settings > Pages**.
4. Em **Source**, selecione **GitHub Actions**.
5. O workflow `.github/workflows/update_data.yml` já está configurado para fazer o deploy automaticamente.

## Atualização automática

O GitHub Actions executa o pipeline nos dias **10 e 13 de cada mês**, quando o IBGE já divulgou o IPCA do mês anterior. Também é possível disparar manualmente em **Actions > Update Dashboard Data > Run workflow**.

O workflow:
1. Instala as dependências.
2. Executa `python -m src.builders.generate_site`.
3. Faz o deploy da pasta `site/` para o GitHub Pages.

## Próximos passos (roadmap)

- [ ] Surpresa do IPCA vs. projeções Focus (BCB)
- [ ] Núcleos de inflação (BCB)
- [ ] CPI dos EUA
- [ ] Comparativo Brasil vs. EUA
- [ ] Tema escuro

## Licença

MIT
