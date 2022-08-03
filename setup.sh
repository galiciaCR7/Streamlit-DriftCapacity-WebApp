mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"galiciacr7@g.ucla.edu\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml