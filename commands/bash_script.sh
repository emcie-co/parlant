export OPENAI_API_KEY=your_api_key
python -m src.parlant.bin.server run --log-level debug

poetry run python -m src.parlant.bin.server run --log-level debug
poetry run python src/parlant/bin/server.py run --log-level debug

poetry run python scripts/run_parlant_server.py


echo 'VITE_BASE_URL=http://localhost:8800' > src/parlant/api/chat/.env.local

npm install
npm run dev

npm run build
npm run preview