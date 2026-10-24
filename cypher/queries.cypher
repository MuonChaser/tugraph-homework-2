// TuGraph 图数据计数核验
MATCH (n:webpage)
RETURN count(n) AS webpage_count;

MATCH ()-[r:hyperlink]->()
RETURN count(r) AS hyperlink_count;

// 查看 PageRank 第一名网页的出链（page_id 来自 results/pagerank_top20.csv）
MATCH (a:webpage {id: 89073})-[:hyperlink]->(b:webpage)
RETURN a.id AS source_page, b.id AS linked_page
LIMIT 20;
