import streamlit as st
import sys
from pathlib import Path
from PIL import Image
import base64
from io import BytesIO

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.dialog import (
    SmartDialogManager,
    LengthTerm, ImportanceTerm, FreshnessTerm,
    TERM_LABELS
)
from src.dialog.linguistic_variable import (
    get_length_membership,
    get_importance_membership,
    get_freshness_membership
)
from src.recommender.hybrid import hybrid_recommendation_search
from src.recommender.search import search_articles, fuzzy_search_articles
from src.data.dataset import NewsDataset
from src.features.similarity_metrics import SimilarityCalculator
from src.recommender.filtering import get_similar_articles
from src.recommender.likes_recommender import recommend_from_likes
from src.recommender.dislikes_recommender import recommend_with_dislikes


def load_image(image_name: str):
    image_path = project_root / "src" / "ui" / "assets" / "images" / image_name

    if image_path.exists():
        try:
            return Image.open(image_path)
        except Exception as e:
            st.warning(f"Не удалось загрузить {image_name}: {e}")
            return None
    else:
        st.warning(f"Файл не найден: {image_path}")
        return None


def load_css():
    css_path = project_root / "src" / "ui" / "assets" / "styles" / "custom.css"

    if css_path.exists():
        with open(css_path, 'r', encoding='utf-8') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.set_page_config(
    page_title="Рекомендательная система новостей",
    page_icon=load_image("news.png"),
    layout="wide",
    initial_sidebar_state="expanded"
)

load_css()

@st.cache_data
def load_data():
    dataset = NewsDataset()
    df = dataset.load()
    df = dataset.preprocess()
    return df


@st.cache_data
def load_similarity_matrices(_df):
    calculator = SimilarityCalculator(_df)

    try:
        calculator.load_all()
        if 'comprehensive' in calculator.similarity_matrices:
            return calculator
    except:
        pass

    with st.spinner('Вычисление матриц сходства... Это может занять минуту.'):
        calculator.compute_comprehensive_similarity()

    return calculator


def display_article(article, compact=False):
    st.write(f"**Категория:** {article['category']}")
    st.write(f"**Автор:** {article['author']}")
    st.write(f"**Теги:** {article['tags']}")

    if not compact:
        st.write(f"**Дата:** {article['published_at']}")
        st.write(f"**Длина:** {article['content_length']} слов")
        st.write(f"**Комментариев:** {article['comment_number']}")

        st.write(f"**Медиа:**")
        if article['has_video']:
            video_icon = load_image("video.png")
            if video_icon:
                buffered = BytesIO()
                video_icon.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()

                st.markdown(
                    f"""
                    <div style="display: flex; align-items: center; margin-bottom: 20px;">
                    <img src="data:image/png;base64,{img_str}" width="20" style="margin-right: 15px;">
                    <h7 style="margin: 0;">Видео</h7>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        if article['has_image']:
            image_icon = load_image("image.png")
            if image_icon:
                buffered = BytesIO()
                image_icon.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                st.markdown(
                    f"""
                    <div style="display: flex; align-items: center; margin-bottom: 20px;">
                    <img src="data:image/png;base64,{img_str}" width="20" style="margin-right: 15px;">
                    <h7 style="margin: 0;">Изображение</h7>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

def page_home(df):
    hot_news = load_image("hot_news.png")
    if hot_news:
            buffered = BytesIO()
            hot_news.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            st.markdown(
                f"""
                <div style="display: flex; align-items: center; margin-bottom: 20px;">
                <img src="data:image/png;base64,{img_str}" width="50" style="margin-right: 15px;">
                <h1 style="margin: 0;">Свежие статьи</h1>
                </div>
                """,
                unsafe_allow_html=True
            )
    latest = df.sort_values('published_at', ascending=False).head(5)

    for idx, row in latest.iterrows():
        with st.expander(f"#{row['article_id']}: {row['category']}"):
            display_article(row)


def page_similar(df, similarity_df):
    article_ids = df['article_id'].tolist()

    col1, col2 = st.columns([3, 1])

    with col1:
        selected_id = st.selectbox(
            "Выберите статью",
            article_ids,
            format_func=lambda x: f"#{x}: {df[df['article_id'] == x].iloc[0]['category']}"
        )

    with col2:
        n_recommendations = st.number_input(
            "Количество",
            min_value=1,
            max_value=20,
            value=5
        )

    if st.button("Найти похожие", type="primary"):
        original = df[df['article_id'] == selected_id].iloc[0]

        st.markdown("---")
        st.subheader("Исходная статья")
        display_article(original)

        similar = get_similar_articles(selected_id, similarity_df, n=n_recommendations)

        st.markdown("---")
        st.subheader(f"Топ-{n_recommendations} похожих статей")

        for rank, (sim_id, score) in enumerate(similar.items(), 1):
            article = df[df['article_id'] == sim_id].iloc[0]

            with st.expander(f"{rank}. [ID:{sim_id}] Сходство: {score:.3f}"):
                display_article(article)


def page_likes(df, similarity_df):
    article_ids = df['article_id'].tolist()

    liked_ids = st.multiselect(
        "Выберите статьи, которые вам понравились",
        article_ids,
        format_func=lambda x: f"#{x}: {df[df['article_id'] == x].iloc[0]['category']}"
    )

    col1, col2 = st.columns(2)

    with col1:
        n_recommendations = st.number_input(
            "Количество рекомендаций",
            min_value=1,
            max_value=50,
            value=10
        )

    with col2:
        aggregation = st.selectbox(
            "Метод агрегации",
            ["mean", "max"],
            format_func=lambda x: {"mean": "Среднее", "max": "Максимум"}[x]
        )

    if st.button("Получить рекомендации", type="primary"):
        if not liked_ids:
            st.error("Выберите хотя бы одну статью!")
            return

        st.markdown("---")
        st.subheader("Вам понравилось:")

        for aid in liked_ids:
            article = df[df['article_id'] == aid].iloc[0]
            with st.expander(f"#{aid}: {article['category']}"):
                display_article(article, compact=True)

        try:
            recommendations = recommend_from_likes(
                liked_articles=liked_ids,
                similarity_df=similarity_df,
                df=df,
                n=n_recommendations,
                aggregation=aggregation
            )

            st.markdown("---")
            st.subheader(f"Топ-{n_recommendations} рекомендаций")

            for idx, row in recommendations.iterrows():
                with st.expander(f"{row['rank']}. [ID:{row['article_id']}] Рейтинг: {row['score']:.4f}"):
                    article = df[df['article_id'] == row['article_id']].iloc[0]
                    display_article(article)

                    st.markdown("**Сходство с лайками:**")
                    for liked_id, sim in row['similarities_to_liked'].items():
                        st.write(f"  - С #{liked_id}: {sim:.3f}")

        except Exception as e:
            st.error(f"Ошибка: {e}")


def page_likes_dislikes(df, similarity_df):
    article_ids = df['article_id'].tolist()

    col1, col2 = st.columns(2)

    with col1:
        liked_ids = st.multiselect(
            "Понравилось",
            article_ids,
            format_func=lambda x: f"#{x}: {df[df['article_id'] == x].iloc[0]['category']}",
            key="likes_dis"
        )

    with col2:
        disliked_ids = st.multiselect(
            "Не понравилось",
            article_ids,
            format_func=lambda x: f"#{x}: {df[df['article_id'] == x].iloc[0]['category']}",
            key="dislikes"
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        n_recommendations = st.number_input("Количество", 1, 50, 10)

    with col2:
        like_weight = st.slider("Вес лайков", 0.0, 2.0, 1.0, 0.1)

    with col3:
        dislike_weight = st.slider("Вес дизлайков", 0.0, 2.0, 0.5, 0.1)

    if st.button("Получить рекомендации", type="primary"):
        intersection = set(liked_ids) & set(disliked_ids)
        if intersection:
            st.error(f"Статьи {intersection} присутствуют и в лайках, и в дизлайках")
            return

        if not liked_ids and not disliked_ids:
            st.error("Выберите хотя бы лайки или дизлайки")
            return

        try:
            recommendations = recommend_with_dislikes(
                liked_articles=liked_ids,
                disliked_articles=disliked_ids,
                similarity_df=similarity_df,
                df=df,
                n=n_recommendations,
                like_weight=like_weight,
                dislike_weight=dislike_weight
            )

            st.markdown("---")
            st.subheader(f"Рекомендации")

            for idx, row in recommendations.iterrows():
                score_color = "green" if row['total_score'] > 0 else "red"

                with st.expander(
                        f"{row['rank']}. [ID:{row['article_id']}] Рейтинг: :{score_color}[{row['total_score']:.4f}]"):
                    st.write(f"**Положительный score:** +{row['positive_score']:.4f}")
                    st.write(f"**Отрицательный score:** -{row['negative_score']:.4f}")
                    st.markdown("---")
                    article = df[df['article_id'] == row['article_id']].iloc[0]
                    display_article(article)

        except Exception as e:
            st.error(f"Ошибка: {e}")


def page_parametric_search(df):
    min_len = int(df['content_length'].min()) if 'content_length' in df.columns else 0
    max_len = int(df['content_length'].max()) if 'content_length' in df.columns else 10000

    if 'search_history' not in st.session_state:
        st.session_state.search_history = []

    if 'applied_filters' not in st.session_state:
        st.session_state.applied_filters = {}

    if 'f_cat' not in st.session_state: st.session_state.f_cat = []
    if 'f_auth' not in st.session_state: st.session_state.f_auth = []
    if 'f_tags' not in st.session_state: st.session_state.f_tags = []
    if 'f_len' not in st.session_state: st.session_state.f_len = (min_len, max_len)

    def undo_callback():
        if st.session_state.search_history:
            previous_state = st.session_state.search_history.pop()
            st.session_state.applied_filters = previous_state

            st.session_state.f_cat = previous_state.get('categories', [])
            st.session_state.f_auth = previous_state.get('authors', [])
            st.session_state.f_tags = previous_state.get('tags', [])
            st.session_state.f_len = previous_state.get('length_range', (min_len, max_len))

    with st.expander("Фильтры", expanded=True):
        col1, col2 = st.columns(2)

        all_categories = sorted(df['category'].unique().tolist())
        all_authors = sorted(df['author'].unique().tolist())
        all_tags = set()
        for tags in df['tags']:
            if isinstance(tags, list):
                all_tags.update(tags)

        with col1:
            selected_categories = st.multiselect("Категории", all_categories, key="f_cat")
            len_range = st.slider(
                "Длина статьи (слов)",
                min_value=min_len,
                max_value=max_len,
                key="f_len"
            )

        with col2:
            selected_authors = st.multiselect("Авторы", all_authors, key="f_auth")

        btn_col1, btn_col2, _ = st.columns([1, 1, 3])

        with btn_col1:
            do_search = st.button("Поиск", type="primary", use_container_width=True)

        with btn_col2:
            st.button(
                "↩ Назад",
                disabled=len(st.session_state.search_history) == 0,
                on_click=undo_callback,
                use_container_width=True
            )

    if do_search:
        current_widget_state = {
            'categories': selected_categories,
            'authors': selected_authors,
            'length_range': len_range
        }

        st.session_state.search_history.append(st.session_state.applied_filters.copy())
        st.session_state.applied_filters = current_widget_state

    active_filters = st.session_state.applied_filters

    if not active_filters and not do_search:
        st.info("Задайте параметры выше и нажмите 'Поиск'.")
        return

    st.subheader("Результаты поиска")

    results = search_articles(df, active_filters)

    if not results.empty:
        st.success(f"Найдено: {len(results)}")
        for idx, row in results.iterrows():
            with st.expander(f"#{row['article_id']}: {row['category']} - {row['author']}"):
                display_article(row)
    else:
        st.warning("Точных совпадений не найдено.")
        st.markdown("### Возможно, вам понравится:")

        fuzzy_results = fuzzy_search_articles(df, active_filters, n=5)

        if not fuzzy_results.empty:
            for idx, row in fuzzy_results.iterrows():
                reasons = []
                if 'categories' in active_filters and row['category'] in active_filters['categories']:
                    reasons.append("Категория")
                if 'authors' in active_filters and row['author'] in active_filters['authors']:
                    reasons.append("Автор")

                with st.expander(f"#{row['article_id']}: {row['category']}"):
                    st.caption(f"Совпало: {', '.join(reasons) if reasons else 'Теги/Длина'}")
                    display_article(row)
        else:
            st.error("Ничего похожего не найдено.")


def page_hybrid_system(df, similarity_df):
    if 'hybrid_history' not in st.session_state:
        st.session_state.hybrid_history = []

    if 'hybrid_state' not in st.session_state:
        st.session_state.hybrid_state = {
            'liked': [],
            'disliked': [],
            'filters': {}
        }

    min_len = int(df['content_length'].min()) if 'content_length' in df.columns else 0
    max_len = int(df['content_length'].max()) if 'content_length' in df.columns else 10000

    defaults = {
        'h_liked': [], 'h_disliked': [],
        'h_cat': [], 'h_auth': [], 'h_tags': [],
        'h_len': (min_len, max_len)
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    def undo_hybrid_callback():
        if st.session_state.hybrid_history:
            prev = st.session_state.hybrid_history.pop()
            st.session_state.hybrid_state = prev

            st.session_state.h_liked = prev.get('liked', [])
            st.session_state.h_disliked = prev.get('disliked', [])

            f = prev.get('filters', {})
            st.session_state.h_cat = f.get('categories', [])
            st.session_state.h_auth = f.get('authors', [])
            st.session_state.h_tags = f.get('tags', [])
            st.session_state.h_len = f.get('length_range', (min_len, max_len))

    with st.expander("Настройка поиска и рекомендаций", expanded=True):
        col_prefs1, col_prefs2 = st.columns(2)

        article_ids = df['article_id'].tolist()
        format_func = lambda x: f"#{x}: {df[df['article_id'] == x].iloc[0]['category']}"

        with col_prefs1:
            sel_liked = st.multiselect("Понравилось", article_ids, format_func=format_func, key="h_liked")
        with col_prefs2:
            sel_disliked = st.multiselect("Не понравилось", article_ids, format_func=format_func,
                                          key="h_disliked")

        st.markdown("---")

        col_filt1, col_filt2 = st.columns(2)

        all_categories = sorted(df['category'].unique().tolist())
        all_authors = sorted(df['author'].unique().tolist())

        with col_filt1:
            sel_cat = st.multiselect("Категории", all_categories, key="h_cat")
            sel_len = st.slider("Длина (слов)", min_len, max_len, key="h_len")

        with col_filt2:
            sel_auth = st.multiselect("Авторы", all_authors, key="h_auth")

        st.markdown("---")

        c1, c2, _ = st.columns([1, 1, 3])
        with c1:
            do_run = st.button("Применить", type="primary", use_container_width=True)
        with c2:
            st.button("↩ Отменить действие",
                      disabled=len(st.session_state.hybrid_history) == 0,
                      on_click=undo_hybrid_callback,
                      use_container_width=True)

    if do_run:
        st.session_state.hybrid_history.append(st.session_state.hybrid_state.copy())

        new_filters = {
            'categories': sel_cat,
            'authors': sel_auth,
            'length_range': sel_len
        }

        new_state = {
            'liked': sel_liked,
            'disliked': sel_disliked,
            'filters': new_filters
        }

        st.session_state.hybrid_state = new_state

    curr_state = st.session_state.hybrid_state

    is_empty_request = (
            not curr_state['liked'] and
            not curr_state['disliked'] and
            not any(curr_state['filters'].values())
    )

    if is_empty_request and not do_run:
        st.info("Настройте предпочтения или фильтры выше и нажмите 'Применить'.")
        return

    st.divider()
    st.subheader("Результаты")

    try:
        results = hybrid_recommendation_search(
            df=df,
            similarity_df=similarity_df,
            liked_ids=curr_state['liked'],
            disliked_ids=curr_state['disliked'],
            filters=curr_state['filters'],
            top_n=10
        )

        if results.empty:
            st.warning("Ничего не найдено даже с учетом мягкого поиска.")
        else:
            for idx, row in results.iterrows():
                score_val = row.get('rec_score', 0)
                utility_val = row.get('utility_score', score_val)  # Если utility нет, значит это точный поиск

                score_display = f"{utility_val:.3f}"

                with st.expander(f"#{row['article_id']}: {row['category']} "):
                    col_info, col_metrics = st.columns([3, 1])
                    with col_info:
                        display_article(row)
                    with col_metrics:
                        if 'utility_score' in row:
                            st.caption("Возможно интересно")
                        else:
                            st.caption("Точное соответствие фильтрам")

    except Exception as e:
        st.error(f"Ошибка вычисления: {e}")


def page_dialog_system(df):

    #ИНИЦИАЛИЗАЦИЯ СОСТОЯНИ
    if 'dialog_manager' not in st.session_state:
        with st.spinner("Загрузка нейросетевых моделей..."):
            st.session_state.dialog_manager = SmartDialogManager(df)

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    manager = st.session_state.dialog_manager

    # Выводим все сообщения, накопленные в сессии
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Например: Найди короткие важные новости"):

        # 4.1. Добавляем сообщение пользователя в UI
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 4.2. Получаем ответ от системы
        # spinner нужен, так как первый запуск может занять время (загрузка модели)
        with st.spinner("Анализирую запрос..."):
            response = manager.process(prompt)

        # 4.3. ОБРАБОТКА УПРАВЛЯЮЩИХ СИГНАЛОВ
        if response == "SIGNAL_RESET":
            # Полная очистка
            st.session_state.messages = []
            st.rerun()

        elif response == "SIGNAL_UNDO":
            # Логика отмены в UI:
            # Нам нужно удалить:
            # 1. Текущий запрос пользователя ("назад" или "отмени")
            # 2. Предыдущий ответ бота
            # 3. Предыдущий запрос пользователя
            # Итого 3 элемента с конца.

            if len(st.session_state.messages) >= 3:
                st.session_state.messages.pop()  # Удаляем текущий "назад"
                st.session_state.messages.pop()  # Удаляем пред. ответ бота
                st.session_state.messages.pop()  # Удаляем пред. вопрос юзера
                st.toast("Шаг отменен")
                st.rerun()
            else:
                # Если истории нет, просто удаляем команду "назад" из UI и говорим, что нечего отменять
                st.session_state.messages.pop()
                st.warning("Нечего отменять, это начало диалога.")
                # Не делаем rerun, чтобы warning висел, или делаем rerun без добавления сообщения

        else:
            # 4.4. ОБЫЧНЫЙ ОТВЕТ
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)



def run_manual_query(text, manager):
    # Добавляем вопрос
    st.session_state.messages.append({"role": "user", "content": text})
    # Получаем ответ
    response = manager.process(text)
    # Добавляем ответ (тут сигналы не обрабатываем, т.к. кнопки только для старта)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

def main():

    with st.spinner('Загрузка данных...'):
        df = load_data()

    with st.spinner('Загрузка матриц сходства...'):
        calculator = load_similarity_matrices(df)
        similarity_df = calculator.get_similarity_matrix('comprehensive')


    st.sidebar.title("Навигация")

    page = st.sidebar.radio(
        "Выберите страницу",
        ["Главная",
         "Поиск рекомендаций",
         "Рекомендации по лайкам",
         "Рекомендации по лайкам и дизлайкам",
         "Параметрический поиск",
         "Гибридная система",
         "Диалоговая система"]
    )

    if page == "Главная":
        page_home(df)
    elif page == "Поиск рекомендаций":
        page_similar(df, similarity_df)
    elif page == "Рекомендации по лайкам":
        page_likes(df, similarity_df)
    elif page == "Рекомендации по лайкам и дизлайкам":
        page_likes_dislikes(df, similarity_df)
    elif page == "Параметрический поиск":  # <--- Добавлено
        page_parametric_search(df)
    elif page == "Гибридная система":
        page_hybrid_system(df, similarity_df)
    elif page == "Диалоговая система":  # <-- ДОБАВИТЬ
        page_dialog_system(df)


if __name__ == "__main__":
    main()