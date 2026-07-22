# Inflation Dashboard

Painel de inflação **Brasil + Estados Unidos**, atualizado automaticamente e publicado em:

**https://pedrobolt.github.io/inflation-dashboard/**

Site estático gerado por um pipeline em Python: coleta dados de fontes oficiais, processa e renderiza gráficos interativos (Plotly). Sem backend — tudo roda no GitHub Actions e é servido pelo GitHub Pages.

## O que o painel mostra

### 🇧🇷 Brasil (IPCA)

| Aba | Conteúdo |
|-----|----------|
| **Resumo** | Tabela com heatmap: MoM e YoY do índice geral, categorias (Livres, Administrados, Serviços, Industriais, Alimentação) e núcleos do BCB (EX0, EX3, MS, DP, P55) |
| **Destaques** | Top 10 contribuições positivas e negativas em bps (peso × MoM) + sazonalidade de cada subitem |
| **Surpresa** | Realizado vs. projeções Focus, em bps (1M, 3M, 6M, 12M), por categoria |
| **Por Grupos** | SAAR 1M/3M/6M vs. YoY por grupo de núcleos, sazonalidade, headline vs. média dos núcleos e sub-núcleos individuais |

Fontes: IBGE SIDRA (IPCA e subitens), BCB SGS (núcleos e categorias oficiais), BCB Focus (projeções), planilha de vetores de agregação do BCB (pesos).

### 🇺🇸 Estados Unidos (CPI)

| Aba | Conteúdo |
|-----|----------|
| **Resumo** | Momentum do core CPI (SAAR 1M/3M/6M vs. YoY) e comparação CPI vs. PCE (a meta de 2% do Fed é sobre o PCE) |
| **Composição** | Os "três baldes" que o Fed acompanha: core goods, shelter e supercore, mais food e energy |
| **Shelter** | CPI Shelter vs. aluguel de mercado (Zillow ZORI) — o ZORI antecipa o CPI Shelter em ~12 meses |
| **Expectativas** | Breakevens de TIPS (5A e 5A5A forward), Michigan, e amplitude via median/trimmed-mean/sticky CPI |

Fontes: FRED (St. Louis Fed) e Zillow Research (CSV público).

## Como rodar localmente

```bash
git clone https://github.com/pedrobolt/inflation-dashboard.git
cd inflation-dashboard
pip install -r requirements.txt
```

Para o painel dos EUA, crie uma chave gratuita do FRED em
https://fred.stlouisfed.org/docs/api/api_key.html e salve na raiz do projeto:

```bash
echo "FRED_API_KEY=sua_chave_aqui" > .env
```

(Sem a chave o build funciona normalmente, publicando só o painel Brasil.)

Gere o site e sirva localmente:

```bash
python -m src.builders.generate_site
cd site && python -m http.server 8000
```

Abra http://localhost:8000.

Testes:

```bash
python tests/test_processors.py
```

## Atualização automática

O GitHub Actions roda o pipeline nos dias **10 e 13 de cada mês** (quando IBGE e BLS já divulgaram os índices do mês anterior) e faz o deploy no GitHub Pages. Também dá para disparar manualmente em **Actions → Update Dashboard Data → Run workflow**.

Para o painel EUA no deploy, cadastre a chave em **Settings → Secrets and variables → Actions** como `FRED_API_KEY`.

## Estrutura

```
src/
├── collectors/
│   ├── brazil/     # IBGE SIDRA, BCB SGS, Focus, vetores de agregação
│   └── us/         # FRED, Zillow ZORI
├── processors/
│   ├── brazil/     # Resumo, Destaques, Surpresa, Grupos + geradores de gráfico
│   └── us/         # Painel EUA (reusa os geradores de gráfico)
└── builders/       # Orquestração e template HTML (Jinja2)
site/               # Site gerado (publicado no Pages)
.github/workflows/  # Automação (cron + deploy)
tests/              # Testes dos processadores
```

## Licença

MIT
