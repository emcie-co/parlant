import './App.css';
import Chatbot from './components/chatbot/chatbot';
import {useWebSocket} from './hooks/useWebSocket';
import {BASE_URL} from './utils/api';
import {handleChatLogs} from './utils/logs';

const WebSocketComp = () => {
	const socket = useWebSocket(`${BASE_URL}/logs`, true, null, handleChatLogs);
	void socket;
	return <div></div>;
};

function App() {
	return (
		<div className='bg-green-light'>
			<Chatbot />
			<WebSocketComp />
		</div>
	);
}

export default App;
