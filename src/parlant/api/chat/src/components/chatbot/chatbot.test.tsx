import { describe, expect, it, vi } from 'vitest';
import { MatcherOptions, render } from '@testing-library/react';
import { Matcher } from 'vite';

import Chatbot from './chatbot';

vi.mock('../virtual-scroll/virtual-scroll', () => ({
    default: vi.fn(({children}) => <div>{children}</div>)
}));

describe(Chatbot, () => {
    let getByTestId: (id: Matcher, options?: MatcherOptions | undefined) => HTMLElement;
    
    beforeEach(() => {
        const utils = render(<Chatbot/>);
        getByTestId = utils.getByTestId as (id: Matcher, options?: MatcherOptions | undefined) => HTMLElement;
    });

    it('component should be rendered', () => {
        const submitButton = getByTestId('chatbot');
        expect(submitButton).toBeInTheDocument();
    });
});