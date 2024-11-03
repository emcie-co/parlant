import { createContext, Dispatch, lazy, ReactElement, ReactNode, SetStateAction, Suspense, useContext, useState } from 'react';
import { AgentInterface, SessionInterface } from '@/utils/interfaces';
import Sessions from '../sessions/sessions';
import ErrorBoundary from '../error-boundary/error-boundary';
import ChatHeader from '../chat-header/chat-header';
import { Dimensions, useDialog } from '@/hooks/useDialog';

interface SessionContext {
    setSessionId: Dispatch<SetStateAction<string | null | undefined>>;
    sessionId: string | null | undefined;
    setAgentId: Dispatch<SetStateAction<string | null>>;
    agentId: string | null;
    setNewSession: Dispatch<SetStateAction<SessionInterface | null>>;
    newSession: SessionInterface | null;
    sessions: SessionInterface[],
    setSessions: Dispatch<SetStateAction<SessionInterface[]>>;
    agents: AgentInterface[],
    setAgents: Dispatch<SetStateAction<AgentInterface[]>>;
    openDialog: (title: string, content: ReactNode, dimensions: Dimensions, dialogClosed?: (() =>void) | null) => void;
    closeDialog: () => void;
};

export const SessionProvider = createContext<SessionContext>({
    sessionId: null,
    setSessionId: () => null,
    agentId: null,
    setAgentId: () => null,
    newSession: null,
    setNewSession: () => null,
    sessions: [],
    setSessions: () => null,
    agents: [],
    setAgents: () => null,
    openDialog: () => null,
    closeDialog: () =>null
});

// eslint-disable-next-line react-refresh/only-export-components
export const useSession = () => useContext(SessionProvider);

export default function Chatbot(): ReactElement {
    const Chat = lazy(() => import('../chat/chat'));
    const [sessionId, setSessionId] = useState<string | null | undefined>(null);
    const [sessions, setSessions] = useState<SessionInterface[]>([]);
    const [agentId, setAgentId] = useState<string | null>(null);
    const [newSession, setNewSession] = useState<SessionInterface | null>(null);
    const [agents, setAgents] = useState<AgentInterface[]>([]);
    const {openDialog, DialogComponent, closeDialog} = useDialog();

    const provideObj = {
        sessionId,
        setSessionId,
        agentId,
        setAgentId,
        newSession,
        setNewSession,
        sessions,
        setSessions,
        agents,
        setAgents,
        openDialog,
        closeDialog
    };

    return (
        <ErrorBoundary>
            <SessionProvider.Provider value={provideObj}>
                <div data-testid="chatbot" className="main bg-main h-screen flex flex-col">
                    <ChatHeader/>
                    <div className="flex justify-between flex-1 w-full overflow-auto flex-row">
                        <div className="bg-white h-full pb-4 border-solid w-[332px] max-mobile:hidden">
                            <Sessions />
                        </div>
                        <div className='h-full w-full'>
                            {sessionId && <Suspense><Chat /></Suspense>}
                        </div>
                    </div>
                </div>
                <DialogComponent />
            </SessionProvider.Provider>
        </ErrorBoundary>
    );
}