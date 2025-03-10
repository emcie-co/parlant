import {createContext, ReactElement, useEffect, useState} from 'react';
import SessionList from '../session-list/session-list';
import ErrorBoundary from '../error-boundary/error-boundary';
import ChatHeader from '../chat-header/chat-header';
import {useDialog} from '@/hooks/useDialog';
import {Helmet} from 'react-helmet';
import {NEW_SESSION_ID} from '../agents-list/agent-list';
import {useAtom} from 'jotai';
import {dialogAtom, sessionAtom} from '@/store';
import {twMerge} from 'tailwind-merge';
import SessionView from '../session-view/session-view';

export const SessionProvider = createContext({});

const SessionsSection = () => {
	const [filterSessionVal, setFilterSessionVal] = useState('');
	return (
		<div className='bg-white [box-shadow:0px_0px_25px_0px_#0000000A] h-full rounded-[16px] overflow-hidden border-solid w-[352px] max-mobile:hidden z-[11] '>
			<ChatHeader setFilterSessionVal={setFilterSessionVal} />
			<SessionList filterSessionVal={filterSessionVal} />
		</div>
	);
};

export default function Chatbot(): ReactElement {
	// const SessionView = lazy(() => import('../session-view/session-view'));
	const [sessionName, setSessionName] = useState<string | null>('');
	const {openDialog, DialogComponent, closeDialog} = useDialog();
	const [session] = useAtom(sessionAtom);
	const [, setDialog] = useAtom(dialogAtom);
	const [, setFilterSessionVal] = useState('');

	useEffect(() => {
		if (session?.id) {
			if (session?.id === NEW_SESSION_ID) setSessionName('Parlant | New Session');
			else {
				const sessionTitle = session?.title;
				if (sessionTitle) setSessionName(`Parlant | ${sessionTitle}`);
			}
		} else setSessionName('Parlant');
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [session?.id]);

	useEffect(() => {
		setDialog({openDialog, closeDialog});
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, []);

	return (
		<ErrorBoundary>
			<SessionProvider.Provider value={{}}>
				<Helmet defaultTitle={`${sessionName}`} />
				<div className={'flex items-center bg-green-main h-[60px] mb-[14px] [box-shadow:0px_0px_25px_0px_#0000000A]'}>
					<img src='/chat/app-logo.svg' alt='logo' aria-hidden className='ms-[27px] self-center me-[6px] max-mobile:ms-0' />
				</div>
				<div data-testid='chatbot' className={'main bg-green-light h-[calc(100vh-74px)] flex flex-col rounded-[16px]'}>
					<div className='hidden max-mobile:block rounded-[16px]'>
						<ChatHeader setFilterSessionVal={setFilterSessionVal} />
					</div>
					<div className={twMerge('flex bg-green-light justify-between flex-1 gap-[14px] w-full overflow-auto flex-row pb-[14px] px-[14px]')}>
						<SessionsSection />
						{session?.id ? (
							<div className='h-full w-[calc(100vw-352px-28px)] bg-white rounded-[16px] max-w-[calc(100vw-352px-28px)] max-[750px]:max-w-full max-[750px]:w-full '>
								{/* <Suspense> */}
								<SessionView />
								{/* </Suspense> */}
							</div>
						) : (
							<div className='flex-1 flex items-center justify-center'>
								<img src='select-session.svg' fetchPriority='high' alt='' />
							</div>
						)}
					</div>
				</div>
			</SessionProvider.Provider>
			<DialogComponent />
		</ErrorBoundary>
	);
}
