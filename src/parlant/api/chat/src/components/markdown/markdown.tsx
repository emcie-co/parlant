/* eslint-disable @typescript-eslint/no-unused-vars */
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkBreaks from 'remark-breaks';
import rehypeHighlight from 'rehype-highlight';
import rehypeRaw from 'rehype-raw';
import 'highlight.js/styles/github.css';
import styles from '../message/message.module.scss';
import {twMerge} from 'tailwind-merge';

const Markdown = ({children, className}: {children: string; className?: string}) => {
	return (
		<ReactMarkdown
			components={{
				p: 'div',
				img: ({node, ...props}) => <img {...props} loading='lazy' alt='' />
			}}
			rehypePlugins={[rehypeHighlight, rehypeRaw]}
			remarkPlugins={[remarkGfm, remarkBreaks]}
			className={twMerge('leading-[19px]', styles.markdown, className)}>
			{children}
		</ReactMarkdown>
	);
};

export default Markdown;