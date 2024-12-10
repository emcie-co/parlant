import { ReactNode, useEffect, useState } from 'react';
import Tooltip from '../ui/custom/tooltip';
import { useSession } from '../chatbot/chatbot';
import { spaceClick } from '@/utils/methods';
import AgentList from '../agents-list/agent-list';
import { Menu } from 'lucide-react';
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle, SheetTrigger } from '../ui/sheet';
import Sessions from '../sessions/sessions';
// import DarkModeToggle from '../dark-mode-toggle/dark-mode-toggle';

export const NEW_SESSION_ID = 'NEW_SESSION';

const ChatHeader = (): ReactNode => {
    const {setAgentId, openDialog} = useSession();
    const [sheetOpen, setSheetOpen] = useState(false);
    const {sessionId} = useSession();

    useEffect(() => {
        if (sheetOpen) setSheetOpen(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [sessionId]);

    const createNewSession = () => {
        setAgentId(null);
        openDialog('', <AgentList/>, {height: '536px', width: '604px'});
     };

    return (
        <div className='h-[70px] min-h-[70px] flex justify-between bg-white border-b-[0.6px] border-b-solid border-muted'>
            <div className='w-[332px] max-mobile:w-full h-[70px] flex items-center justify-between'>
                <div className='flex items-center min-[751px]:hidden'>
                    <div>
                        <Sheet open={sheetOpen} onOpenChange={() => setSheetOpen(!sheetOpen)}>
                            <SheetTrigger asChild onClick={() => setSheetOpen(true)}>
                                <Menu className='ms-[24px] cursor-pointer'/>
                            </SheetTrigger>
                            <SheetContent side='left' className='w-fit px-0'>
                                <SheetHeader>
                                    <SheetTitle className='text-center'></SheetTitle>
                                    <SheetDescription/>
                                </SheetHeader>
                                    <Sessions/>
                            </SheetContent>
                        </Sheet>
                    </div>
                </div>
                <div className='flex items-center'>
                    <img src="/chat/parlant-bubble-app-logo.svg" alt="logo" aria-hidden height={17.9} width={20.89} className='ms-[24px] me-[6px] max-mobile:ms-0'/>
                    <p className='text-[19.4px] font-bold'>Parlant</p>
                </div>
                <div className='group me-[24px]'>
                    <Tooltip value='New Session' side='right'>
                        <div>
                            <img onKeyDown={spaceClick} onClick={createNewSession} tabIndex={1} role='button' src="icons/add.svg" alt="add session" height={28} width={28} className='cursor-pointer group-hover:hidden'/>
                            <img onKeyDown={spaceClick} onClick={createNewSession} tabIndex={1} role='button' src="icons/add-filled.svg" alt="add session" height={28} width={28} className='cursor-pointer hidden group-hover:block'/>
                        </div>
                    </Tooltip>
                </div>
            </div>
            {/* <div className='w-[332px] h-[70px] flex items-center justify-end me-4'>
                <DarkModeToggle/>
            </div> */}
        </div>
    );
};

export default ChatHeader;