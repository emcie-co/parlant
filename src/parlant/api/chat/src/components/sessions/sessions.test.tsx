import { cleanup, MatcherOptions, render, SelectorMatcherOptions } from '@testing-library/react';
import { describe, expect, it, Mock, vi } from 'vitest';
import { Matcher } from 'vite';
import { useContext } from 'react';
import '@testing-library/jest-dom/vitest';

import useFetch from '@/hooks/useFetch.tsx';
import Sessions from './sessions.tsx';

const sessionsArr = [
    { id: 'session1', title: 'Session One' },
    { id: 'session2', title: 'Session Two' }
];

vi.mock('@/hooks/useFetch', () => ({
    default: vi.fn(() => {
        return {
            data: {sessions: sessionsArr},
            refetch: vi.fn(),
            ErrorTemplate: null,
            loading: false
        };
    })
}));

vi.mock('../virtual-scroll/virtual-scroll', () => ({
    default: vi.fn(({children}) => <div>{children}</div>)
}));

vi.mock('react', async () => {
    const actualReact = await vi.importActual('react');
    return {
        ...actualReact,
        useContext: vi.fn(() => ({sessions: sessionsArr, setSessions: vi.fn()}))
    };
});

describe(Sessions, () => {
    let getByTestId: (id: Matcher, options?: MatcherOptions | undefined) => HTMLElement;
    let getByText: (id: Matcher, options?: SelectorMatcherOptions | undefined) => HTMLElement;
    let sessions: HTMLElement;
    let session: HTMLElement[];
    let rerender: (ui: React.ReactNode) => void;
    
    beforeEach(async() => {
        const utils = render(<Sessions/>);
        getByTestId = utils.getByTestId as (id: Matcher, options?: MatcherOptions | undefined) => HTMLElement;
        getByText = utils.getByText as (id: Matcher, options?: SelectorMatcherOptions | undefined) => HTMLElement;
        rerender = utils.rerender;
        sessions = getByTestId('sessions');
        session = await utils.findAllByTestId('session');
    });

    afterEach(() => cleanup());

    it('component should be rendered', () => {
        expect(sessions).toBeInTheDocument();
    });

    it('component should have 2 sessions', () => {
        expect(session).toHaveLength(2);
    });

    it('component should show a loading indication on loading', () => {
        (useContext as Mock).mockImplementationOnce(() => ({
            sessions: [], setSessions: vi.fn()
        }));
        (useFetch as Mock).mockImplementationOnce(() => ({
            loading: true
        }));
        rerender(<Sessions/>);
        const loading = getByText('loading...');
        expect(loading).toBeInTheDocument();
    });

    it.skip('component should show error when it gets one', () => {
        (useFetch as Mock).mockImplementationOnce(() => ({
            ErrorTemplate: () => <div>error</div>
        }));
        rerender(<Sessions/>);
        const error = getByText('error');
        expect(error).toBeInTheDocument();
    });
});