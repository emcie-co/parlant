import { describe, it, vi } from 'vitest';
import { fireEvent, MatcherOptions, render } from '@testing-library/react';
import { Matcher } from 'vite';

import ChatHeader from './chat-header';

const setSessionFn = vi.fn();
const openDialogFn = vi.fn();
vi.mock('react', async () => {
    const actualReact = await vi.importActual('react');
    return {
        ...actualReact,
        useContext: vi.fn(() => ({openDialog: openDialogFn, setSessionId: setSessionFn, setAgentId: vi.fn(), setNewSession: vi.fn()}))
    };
});

describe(ChatHeader, () => {
    let getAllByRole: (id: Matcher, options?: MatcherOptions | undefined) => HTMLElement[];

    beforeEach(() => {
        const utils = render(<ChatHeader />);
        getAllByRole = utils.getAllByRole as typeof getAllByRole;

        vi.clearAllMocks();
    });

    it('clicking the "add session" button should open the agent selection dialog', async () => {
        const addBtn = getAllByRole('button');
        fireEvent.click(addBtn[0]);
        expect(openDialogFn).toHaveBeenCalled();
    });
});