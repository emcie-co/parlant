/* eslint-disable @typescript-eslint/no-explicit-any */
import {clearIndexedDBData, getIndexedDBSize} from '@/utils/logs';
import {Trash} from 'lucide-react';
import {useEffect, useState} from 'react';
import {toast} from 'sonner';
import {twJoin} from 'tailwind-merge';

const IndexedDBData = ({eventId, logsDeleted}: {eventId?: string; logsDeleted?: () => void}) => {
	const [estimatedDataInMB, setEstimatedDataInMB] = useState<number | null>(null);

	const setData = async () => {
		const estimated = await getIndexedDBSize();
		setEstimatedDataInMB(estimated);
	};

	async function handleClearDataClick() {
		try {
			await clearIndexedDBData();
			setData();
			toast.success('IndexedDB data cleared successfully.');
			logsDeleted?.();
		} catch (e) {
			console.log('Error clearing IndexedDB data', e);
			toast.error('Error clearing IndexedDB data');
		}
	}

	useEffect(() => {
		setData();
	}, [eventId]);

	const dataInMB = estimatedDataInMB ? +estimatedDataInMB.toFixed(1) : null;
	return (
		<div className={twJoin('ps-[10px] text-[11px] flex items-center gap-[5px] z-[1] bg-white absolute bottom-0 w-full', !dataInMB && 'hidden')}>
			<div>The logs use approximately {dataInMB}MB of storage (indexedDB)</div>
			<Trash role='button' onClick={handleClearDataClick} size={13} />
		</div>
	);
};
export default IndexedDBData;
