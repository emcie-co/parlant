import {createContext, lazy, ReactElement, Suspense, useEffect, useState} from 'react';
import SessionList from '../session-list/session-list';
import ErrorBoundary from '../error-boundary/error-boundary';
import ChatHeader from '../chat-header/chat-header';
import {useDialog} from '@/hooks/useDialog';
import {Helmet} from 'react-helmet';
import {NEW_SESSION_ID} from '../agents-list/agent-list';
import HeaderWrapper from '../header-wrapper/header-wrapper';
import {useAtom} from 'jotai';
import {dialogAtom, sessionAtom} from '@/store';

export const SessionProvider = createContext({});

export default function Chatbot(): ReactElement {
	const SessionView = lazy(() => import('../session-view/session-view'));
	const [sessionName, setSessionName] = useState<string | null>('');
	const {openDialog, DialogComponent, closeDialog} = useDialog();
	const [session] = useAtom(sessionAtom);
	const [, setDialog] = useAtom(dialogAtom);
	const [filterSessionVal, setFilterSessionVal] = useState('');

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
	}, []);

	return (
		<ErrorBoundary>
			<SessionProvider.Provider value={{}}>
				<Helmet defaultTitle={`${sessionName}`} />
				<div className='flex items-center bg-green-main h-[50px] mb-[8px]'>
					<img src='/chat/app-logo.svg' alt='logo' aria-hidden className='ms-[24px] me-[6px] max-mobile:ms-0' />
				</div>
				<div data-testid='chatbot' className='main bg-green-light h-[calc(100vh-58px)] flex flex-col rounded-[16px]'>
					<div className='hidden max-mobile:block rounded-[16px]'>
						<ChatHeader setFilterSessionVal={setFilterSessionVal} />
					</div>
					<div className='flex bg-green-light justify-between flex-1 gap-[14px] w-full overflow-auto flex-row pb-[14px] ps-[14px]'>
						<div className='bg-white h-full rounded-[16px] overflow-hidden border-solid w-[352px] max-mobile:hidden z-[11] '>
							<ChatHeader setFilterSessionVal={setFilterSessionVal} />
							<SessionList filterSessionVal={filterSessionVal} />
						</div>
						<div className='h-full w-[calc(100vw-352px-28px)] bg-white rounded-[16px] max-w-[calc(100vw-352px-28px)] max-[750px]:max-w-full max-[750px]:w-full '>
							{session?.id ? (
								<Suspense>
									<SessionView />
								</Suspense>
							) : (
								<HeaderWrapper />
							)}
						</div>
					</div>
				</div>
			</SessionProvider.Provider>
			<DialogComponent />
		</ErrorBoundary>
	);
}
