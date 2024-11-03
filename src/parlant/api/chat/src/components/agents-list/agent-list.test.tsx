import { describe, expect, it, Mock, vi } from 'vitest';
import { cleanup, fireEvent, render } from '@testing-library/react';
import { useContext } from 'react';
import '@testing-library/jest-dom/vitest';

import { AgentInterface } from '@/utils/interfaces';
import AgentList from './agent-list';
import { NEW_SESSION_ID } from '../chat-header/chat-header';

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

describe(AgentList, () => {
    afterEach(() => cleanup());
    
    it('dialog should show agents list', async () => {
        const {getByTestId} = render(<AgentList />);
        const agent = getByTestId('agent');
        expect(agent).toBeInTheDocument();
    });

    it('selecting an agent should set the agentId', async () => {
        const {getByTestId} = render(<AgentList />);
        const agent = getByTestId('agent');
        fireEvent.click(agent);
        expect(setAgentIdFn).toBeCalledWith(agents[0].id);
    });

    it('selecting an agent should close the dialog', async () => {
        const {getByTestId} = render(<AgentList />);
        const agent = getByTestId('agent');
        fireEvent.click(agent);
        expect(closeDialogFn).toHaveBeenCalled();
    });

    it('dialog should be closed when creating a new session', async () => {
        (useContext as Mock).mockImplementation(() => ({
            sessionId: null,
            setAgents: vi.fn()
        }));
        const {findByTestId} = render(<AgentList />);
        const dialogContent = await findByTestId('dialog-content').catch(() => null);
        expect(dialogContent).toBeNull();
    });
});