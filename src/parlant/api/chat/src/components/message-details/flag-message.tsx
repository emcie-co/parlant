import {EventInterface} from '@/utils/interfaces';
import {Textarea} from '../ui/textarea';
import {Button} from '../ui/button';
import {dialogAtom} from '@/store';
import {useAtom} from 'jotai';
import {addItemToIndexedDB, deleteItemFromIndexedDB} from '@/lib/utils';
import {useState} from 'react';

interface FlagMessageProps {
	events: EventInterface[];
	sessionId: string;
	onFlag?: (flagValue: string) => void;
	existingFlagValue?: string;
}

const FlagMessage = ({events, sessionId, existingFlagValue, onFlag}: FlagMessageProps) => {
	const [dialog] = useAtom(dialogAtom);
	const [flagValue, setFlagValue] = useState(existingFlagValue || '');
	const correlationId = events?.[0]?.correlation_id;

	const flagMessage = async () => {
		await addItemToIndexedDB('Parlant-flags', 'message_flags', correlationId, {sessionId, correlationId: correlationId, flagValue: flagValue || 'This message is flagged'}, 'update', {name: 'sessionIndex', keyPath: 'sessionId'});
		onFlag?.(flagValue || '');
		dialog.closeDialog();
	};

	const unflagMessage = async () => {
		await deleteItemFromIndexedDB('Parlant-flags', 'message_flags', correlationId);
		onFlag?.('');
		dialog.closeDialog();
	};

	return (
		<div className='px-[24px] pb-3 flex flex-col gap-3 h-full'>
			<div>
				<p className='text-[16px] text-[#959595]'>Feedback provided here will show up in the session's exported CSV file.</p>
			</div>
			<div className='flex flex-col gap-1 mt-[26px] overflow-auto'>
				{events.map((event) => {
					return (
						<div className='message-bubble [&>*]:w-full [&_*]:cursor-default'>
							<div className='px-[22px] py-[20px] bg-[#F5F9F7] rounded-[22px] mb-[10px] !w-fit max-w-[90%]'>{event?.data?.message}</div>
						</div>
					);
				})}
			</div>
			<Textarea autoFocus placeholder='Enter your flag reason' value={flagValue} onChange={(e) => setFlagValue(e.target.value)} className='!ring-0 !ring-offset-0 flex-1 !resize-none text-[16px] placeholder:text-[#959595]' />
			<div className='flex justify-end gap-3'>
				<Button variant='outline' onClick={() => dialog.closeDialog()}>
					Cancel
				</Button>
				{existingFlagValue && (
					<Button variant='outline' onClick={unflagMessage}>
						Unflag
					</Button>
				)}
				<Button className='bg-green-main hover:bg-[#005C3F]' onClick={flagMessage}>
					Save
				</Button>
			</div>
		</div>
	);
};

export default FlagMessage;
