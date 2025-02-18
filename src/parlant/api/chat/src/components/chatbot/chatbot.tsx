import {createContext, lazy, ReactElement, Suspense, useEffect, useState} from 'react';
import Sessions from '../sessions/sessions';
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
	const Chat = lazy(() => import('../chat/chat'));
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
				<div data-testid='chatbot' className='main bg-main h-screen flex flex-col rounded-[16px]'>
					<div className='flex items-center bg-[#f4f5f9] pt-[16px]'>
						<img src='/chat/parlant-bubble-app-logo.svg' alt='logo' aria-hidden height={25} width={30} className='ms-[24px] me-[6px] max-mobile:ms-0' />
						<p className='text-[26.96px] font-bold'>Parlant</p>
					</div>
					<div className='hidden max-mobile:block rounded-[16px]'>
						<ChatHeader setFilterSessionVal={setFilterSessionVal} />
					</div>
					<div className='flex justify-between flex-1 gap-[14px] w-full overflow-auto flex-row p-[14px] bg-[#f4f5f9]'>
						<div className='bg-white h-full pb-4 rounded-[16px] border-solid w-[352px] max-mobile:hidden z-[11] border-e'>
							<ChatHeader setFilterSessionVal={setFilterSessionVal} />
							<Sessions filterSessionVal={filterSessionVal} />
						</div>
						<div className='h-full w-[calc(100vw-332px)] bg-white rounded-[16px] max-w-[calc(100vw-332px)] max-[750px]:max-w-full max-[750px]:w-full '>
							{session?.id ? (
								<Suspense>
									<Chat />
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
