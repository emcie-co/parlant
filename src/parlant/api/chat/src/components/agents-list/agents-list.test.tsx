import { describe, expect, it, Mock, vi } from 'vitest';
import { cleanup, fireEvent, render } from '@testing-library/react';
import '@testing-library/jest-dom/vitest';
import AgentsList from './agents-list';
import { NEW_SESSION_ID } from '../chat-header/chat-header';
import { useContext } from 'react';
import { AgentInterface } from '@/utils/interfaces';

const agents: AgentInterface[] = [{id: 'john', name: 'John'}];

const setAgentIdFn = vi.fn();
const closeDialogFn = vi.fn();
vi.mock('react', async () => {
    const actualReact = await vi.importActual('react');
    return {
        ...actualReact,
        useContext: vi.fn(() => ({
            sessionId: NEW_SESSION_ID,
            setSessionId: vi.fn(),
            setAgents: vi.fn(),
            setNewSession: vi.fn(),
            setAgentId: setAgentIdFn,
            closeDialog: closeDialogFn,
            agents
        }))
    };
});

describe(AgentsList, () => {
    afterEach(() => cleanup());
    
    it('dialog should show agents list', async () => {
        const {getByTestId} = render(<AgentsList />);
        const agent = getByTestId('agent');
        expect(agent).toBeInTheDocument();
    });

    it('selecting an agent should set the agentId', async () => {
        const {getByTestId} = render(<AgentsList />);
        const agent = getByTestId('agent');
        fireEvent.click(agent);
        expect(setAgentIdFn).toBeCalledWith(agents[0].id);
    });

    it('selecting an agent should close the dialog', async () => {
        const {getByTestId} = render(<AgentsList />);
        const agent = getByTestId('agent');
        fireEvent.click(agent);
        expect(closeDialogFn).toHaveBeenCalled();
    });

    it('dialog should be closed when creating a new session', async () => {
        (useContext as Mock).mockImplementation(() => ({
            sessionId: null,
            setAgents: vi.fn()
        }));
        const {findByTestId} = render(<AgentsList />);
        const dialogContent = await findByTestId('dialog-content').catch(() => null);
        expect(dialogContent).toBeNull();
    });
});