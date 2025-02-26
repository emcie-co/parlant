import {EventInterface} from '@/utils/interfaces';
import {Drawer, DrawerContent, DrawerDescription, DrawerHeader, DrawerTitle} from '../ui/drawer';
import MessageDetails from './message-details';
import {memo} from 'react';

interface Props {
	showLogsForMessage: EventInterface | null;
	setShowLogsForMessage: (event: EventInterface | null) => void;
	messagesRef: React.RefObject<HTMLDivElement>;
	regenerateMessageDialog: (index: number) => () => void;
	resendMessageDialog: (index: number) => () => void;
}

const MessageDetailsDrawer = ({showLogsForMessage, setShowLogsForMessage, messagesRef, regenerateMessageDialog, resendMessageDialog}: Props) => {
	return (
		<Drawer modal={false} direction='right' open={!!showLogsForMessage} onClose={() => setShowLogsForMessage(null)}>
			<DrawerContent className='left-[unset] h-full right-0 bg-white' style={{width: `${(messagesRef?.current?.clientWidth || 1) / 2}px`}}>
				<DrawerHeader>
					<DrawerTitle hidden></DrawerTitle>
					<DrawerDescription hidden></DrawerDescription>
				</DrawerHeader>
				<MessageDetails
					event={showLogsForMessage}
					regenerateMessageFn={showLogsForMessage?.index ? regenerateMessageDialog(showLogsForMessage.index) : undefined}
					resendMessageFn={showLogsForMessage?.index || showLogsForMessage?.index === 0 ? resendMessageDialog(showLogsForMessage.index) : undefined}
					closeLogs={() => setShowLogsForMessage(null)}
				/>
			</DrawerContent>
		</Drawer>
	);
};

export default memo(MessageDetailsDrawer);
