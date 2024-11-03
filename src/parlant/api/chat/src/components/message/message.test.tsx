import { describe, expect, it } from 'vitest';
import { MatcherOptions, render } from '@testing-library/react';
import { Matcher } from 'vite';

import { EventInterface, ServerStatus } from '@/utils/interfaces';
import Message from './message';

const serverStatuses: ServerStatus[] = ['pending', 'error', 'accepted', 'acknowledged', 'processing', 'typing', 'ready'];

const event: EventInterface = {
    correlation_id: '',
    creation_utc: new Date(),
    data: {message: 'Hi'},
    kind: 'message',
    offset: 0,
    serverStatus: 'pending',
    source: 'end_user'
};

describe(Message, () => {
    let getByTestId: (id: Matcher, options?: MatcherOptions | undefined) => HTMLElement;
    let rerender: (ui: React.ReactNode) => void;
    
    beforeEach(() => {
        const utils = render(<Message isContinual={false} event={event}/>);
        getByTestId = utils.getByTestId as (id: Matcher, options?: MatcherOptions | undefined) => HTMLElement;
        rerender = utils.rerender;
    });

    it('component should be rendered', () => {
        const message = getByTestId('message');
        expect(message).toBeInTheDocument();
    });

    it('message has the valid icon', () => {
        for (const serverStatus of serverStatuses) {
            rerender(<Message isContinual={false} event={{...event, serverStatus}}/>);
            const icon = getByTestId(serverStatus);
            expect(icon).toBeInTheDocument();
        }
    });
});