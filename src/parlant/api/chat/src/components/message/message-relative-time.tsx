/* eslint-disable react-hooks/exhaustive-deps */
import {timeAgo} from '@/lib/utils';
import {EventInterface} from '@/utils/interfaces';
import {useEffect, useRef, useState} from 'react';

const MessageRelativeTime = ({event}: {event: EventInterface}) => {
	const [time, setTime] = useState(event.serverStatus === 'pending' ? 'Just Now' : timeAgo(event.creation_utc));
	const intervalRef = useRef<NodeJS.Timeout | null>(null);

	const setMessageRelativeTime = () => {
		if (intervalRef.current) clearInterval(intervalRef.current);

		const updateTime = () => setTime(timeAgo(event.creation_utc));
		setTime(event.serverStatus === 'pending' ? 'Just Now' : timeAgo(event.creation_utc));

		const creationDate = new Date(event.creation_utc);
		const now = new Date();
		const diffMinutes = Math.floor((now.getTime() - creationDate.getTime()) / (1000 * 60));

		if (diffMinutes < 60) {
			intervalRef.current = setInterval(updateTime, 60000);
		}
		// else if (diffMinutes < 1440) {
		// 	const minutesPastHour = creationDate.getMinutes();
		// 	const nextFullHour = new Date(now);
		// 	nextFullHour.setMinutes(60 - minutesPastHour, 0, 0);
		// 	const timeUntilNextHour = nextFullHour.getTime() - now.getTime();

		// 	intervalRef.current = setTimeout(() => {
		// 		updateTime();
		// 		intervalRef.current = setInterval(updateTime, 3600000);
		// 	}, timeUntilNextHour);
		// } else {
		// 	const nextMidnight = new Date(now);
		// 	nextMidnight.setHours(24, 0, 0, 0);
		// 	const timeUntilMidnight = nextMidnight.getTime() - now.getTime();
		// 	intervalRef.current = setTimeout(updateTime, timeUntilMidnight);
		// }

		return () => {
			if (intervalRef.current) clearInterval(intervalRef.current);
		};
	};

	useEffect(() => {
		if (event.serverStatus !== 'pending' && time === 'Just Now') setTime(timeAgo(event.creation_utc));
	}, [event?.serverStatus]);

	useEffect(setMessageRelativeTime, [event.creation_utc]);

	return <div className='text-[14px] text-[#A9A9A9] font-light'>{time}</div>;
};

export default MessageRelativeTime;
