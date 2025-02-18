import {ReactNode, useEffect, useState} from 'react';
import Tooltip from '../ui/custom/tooltip';
import {spaceClick} from '@/utils/methods';
import AgentList from '../agents-list/agent-list';
import {Menu} from 'lucide-react';
import {Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle, SheetTrigger} from '../ui/sheet';
import Sessions from '../session-list/session-list';
import HeaderWrapper from '../header-wrapper/header-wrapper';
import {useAtom} from 'jotai';
import {agentAtom, dialogAtom, sessionAtom} from '@/store';
// import DarkModeToggle from '../dark-mode-toggle/dark-mode-toggle';

export const NEW_SESSION_ID = 'NEW_SESSION';

const ChatHeader = (): ReactNode => {
	const [sheetOpen, setSheetOpen] = useState(false);
	const [session, setSession] = useAtom(sessionAtom);
	const [, setAgent] = useAtom(agentAtom);
	const [dialog] = useAtom(dialogAtom);

	useEffect(() => {
		if (sheetOpen) setSheetOpen(false);
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [session]);

	const createNewSession = () => {
		setSession(null);
		setAgent(null);
		dialog.openDialog('', <AgentList />, {height: '536px', width: '604px'});
	};

	return (
		<HeaderWrapper className='z-60 overflow-visible'>
			<div className='w-[332px] border-b boder-b-[0.6px] border-b-[#ebecf0] max-mobile:w-full h-[70px] flex items-center justify-between bg-white'>
				<div className='flex items-center min-[751px]:hidden'>
					<div>
						<Sheet open={sheetOpen} onOpenChange={() => setSheetOpen(!sheetOpen)}>
							<SheetTrigger asChild onClick={() => setSheetOpen(true)}>
								<Menu className='ms-[24px] cursor-pointer' />
							</SheetTrigger>
							<SheetContent side='left' className='w-fit px-0'>
								<SheetHeader>
									<SheetTitle className='text-center'></SheetTitle>
									<SheetDescription />
								</SheetHeader>
								<Sessions />
							</SheetContent>
						</Sheet>
					</div>
				</div>
				<div className='flex items-center'>
					<img src='/chat/parlant-bubble-app-logo.svg' alt='logo' aria-hidden height={17.9} width={20.89} className='ms-[24px] me-[6px] max-mobile:ms-0' />
					<p className='text-[19.4px] font-bold'>Parlant</p>
				</div>
				<div className='group me-[24px]'>
					<Tooltip value='New Session' side='right'>
						<div>
							<img onKeyDown={spaceClick} onClick={createNewSession} tabIndex={1} role='button' src='icons/add.svg' alt='add session' height={28} width={28} className='cursor-pointer group-hover:hidden' />
							<img onKeyDown={spaceClick} onClick={createNewSession} tabIndex={1} role='button' src='icons/add-filled.svg' alt='add session' height={28} width={28} className='cursor-pointer hidden group-hover:block' />
						</div>
					</Tooltip>
				</div>
			</div>
			{/* <div className='flex-1 flex'>
				<div className='bg-red-300 flex-1'>A</div>
				<div className='bg-transparent flex-1'></div>
			</div> */}
			{/* <div className='w-[332px] h-[70px] flex items-center justify-end me-4'>
                <DarkModeToggle/>
            </div> */}
		</HeaderWrapper>
	);
};

export default ChatHeader;
