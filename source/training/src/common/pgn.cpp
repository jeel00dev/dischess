#include "pgn.hpp"
#include <cctype>
#include <cstdlib>

namespace dis {
using namespace obs;

int PgnGame::elo(const char *key) const {
    auto v = tag(key);
    if (v.empty() || !std::isdigit(static_cast<unsigned char>(v[0])))
        return 0;
    return std::atoi(v.c_str());
}

std::optional<int> PgnGame::white_result() const {
    auto r = tag("Result");
    if (r == "1-0")
        return 1;
    if (r == "0-1")
        return -1;
    if (r == "1/2-1/2")
        return 0;
    return std::nullopt;
}

namespace {
bool parse_tag(const std::string &line, std::string &key, std::string &value) {
    // [Key "Value"] with \" escapes inside the value.
    if (line.size() < 4 || line[0] != '[')
        return false;
    size_t sp = line.find(' ');
    size_t q1 = line.find('"');
    if (sp == std::string::npos || q1 == std::string::npos || q1 < sp)
        return false;
    key = line.substr(1, sp - 1);
    value.clear();
    for (size_t i = q1 + 1; i < line.size(); ++i) {
        if (line[i] == '\\' && i + 1 < line.size()) {
            value += line[++i];
            continue;
        }
        if (line[i] == '"')
            return true;
        value += line[i];
    }
    return false;
}
} // namespace

bool PgnReader::next(PgnGame &game) {
    game = PgnGame{};
    std::string line;
    bool in_moves = false, any = false;
    auto get = [&](std::string &out) -> bool {
        if (has_pending_) {
            out = std::move(pending_);
            has_pending_ = false;
            return true;
        }
        if (!std::getline(in_, out))
            return false;
        if (!out.empty() && out.back() == '\r')
            out.pop_back();
        return true;
    };
    while (get(line)) {
        if (!in_moves) {
            if (line.empty())
                continue;
            if (line[0] == '[') {
                std::string k, v;
                if (parse_tag(line, k, v))
                    game.tags[k] = v;
                game.raw += line;
                game.raw += '\n';
                any = true;
                continue;
            }
            in_moves = true;
            game.raw += '\n';
        }
        if (line.empty()) {
            // Blank line terminates movetext once we have some.
            if (!game.movetext.empty())
                break;
            continue;
        }
        if (line[0] == '[' && !game.movetext.empty()) {
            // Missing blank line between games: hand the tag to the next call.
            pending_ = line;
            has_pending_ = true;
            break;
        }
        game.movetext += line;
        game.movetext += ' ';
        game.raw += line;
        game.raw += '\n';
        any = true;
    }
    if (!any)
        return false;
    ++games_;
    return true;
}

std::vector<std::string> san_tokens(const std::string &text) {
    std::vector<std::string> out;
    int variation = 0;
    size_t i = 0, n = text.size();
    while (i < n) {
        char c = text[i];
        if (c == '{') { // comment
            size_t e = text.find('}', i);
            i = e == std::string::npos ? n : e + 1;
            continue;
        }
        if (c == ';') { // rest-of-line comment (already joined; skip to space)
            while (i < n && text[i] != '\n')
                ++i;
            continue;
        }
        if (c == '(') {
            ++variation;
            ++i;
            continue;
        }
        if (c == ')') {
            variation = std::max(0, variation - 1);
            ++i;
            continue;
        }
        if (std::isspace(static_cast<unsigned char>(c))) {
            ++i;
            continue;
        }
        size_t j = i;
        while (j < n && !std::isspace(static_cast<unsigned char>(text[j])) && text[j] != '{' &&
               text[j] != '(' && text[j] != ')')
            ++j;
        std::string tok = text.substr(i, j - i);
        i = j;
        if (variation)
            continue;
        if (tok[0] == '$')
            continue;
        if (tok == "1-0" || tok == "0-1" || tok == "1/2-1/2" || tok == "*")
            continue;
        // strip move numbers "12." / "12..." possibly glued to the move ("12.e4")
        size_t k = 0;
        while (k < tok.size() && std::isdigit(static_cast<unsigned char>(tok[k])))
            ++k;
        if (k && k < tok.size() && tok[k] == '.') {
            while (k < tok.size() && tok[k] == '.')
                ++k;
            tok = tok.substr(k);
        } else if (k == tok.size()) {
            continue;
        }
        while (!tok.empty() && (tok.back() == '+' || tok.back() == '#' || tok.back() == '!' ||
                                tok.back() == '?'))
            tok.pop_back();
        if (!tok.empty())
            out.push_back(tok);
    }
    return out;
}

namespace {
int piece_from_letter(char c) {
    switch (c) {
    case 'N': return KNIGHT;
    case 'B': return BISHOP;
    case 'R': return ROOK;
    case 'Q': return QUEEN;
    case 'K': return KING;
    default: return -1;
    }
}
} // namespace

Move parse_san(Board &board, const std::string &input) {
    std::string san = input;
    while (!san.empty() && (san.back() == '+' || san.back() == '#' || san.back() == '!' || san.back() == '?'))
        san.pop_back();
    if (san.empty())
        return 0;
    MoveList legal;
    board.legal_moves(legal);
    if (san == "O-O" || san == "0-0" || san == "O-O-O" || san == "0-0-0") {
        const bool long_side = san.size() == 5;
        for (auto m : legal)
            if ((m & CASTLE) && (to(m) < from(m)) == long_side)
                return m;
        return 0;
    }
    int piece = PAWN;
    size_t pos = 0;
    if (int p = piece_from_letter(san[0]); p >= 0) {
        piece = p;
        pos = 1;
    }
    int promo = 0;
    if (auto eq = san.find('='); eq != std::string::npos && eq + 1 < san.size()) {
        promo = piece_from_letter(san[eq + 1]);
        san = san.substr(0, eq);
    } else if (piece == PAWN && san.size() >= 3 && piece_from_letter(san.back()) > 0 &&
               std::isdigit(static_cast<unsigned char>(san[san.size() - 2]))) {
        promo = piece_from_letter(san.back()); // "e8Q" form
        san.pop_back();
    }
    if (promo < 0)
        return 0;
    if (san.size() < pos + 2)
        return 0;
    const char tf = san[san.size() - 2], tr = san[san.size() - 1];
    if (tf < 'a' || tf > 'h' || tr < '1' || tr > '8')
        return 0;
    const int target = (tr - '1') * 8 + (tf - 'a');
    int dis_file = -1, dis_rank = -1;
    for (size_t i = pos; i + 2 < san.size(); ++i) {
        char c = san[i];
        if (c >= 'a' && c <= 'h')
            dis_file = c - 'a';
        else if (c >= '1' && c <= '8')
            dis_rank = c - '1';
    }
    Move found = 0;
    int matches = 0;
    for (auto m : legal) {
        if (to(m) != target || promotion(m) != promo)
            continue;
        const int f = from(m);
        if (type(board.squares[f]) != piece || (m & CASTLE))
            continue;
        if (dis_file >= 0 && f % 8 != dis_file)
            continue;
        if (dis_rank >= 0 && f / 8 != dis_rank)
            continue;
        found = m;
        ++matches;
    }
    return matches == 1 ? found : 0;
}

std::string to_san(Board &board, Move m) {
    if (m & CASTLE)
        return to(m) > from(m) ? "O-O" : "O-O-O";
    const int f = from(m), t = to(m), p = type(board.squares[f]);
    std::string s;
    auto square = [](int x) { return std::string{char('a' + x % 8), char('1' + x / 8)}; };
    if (p == PAWN) {
        if (m & CAPTURE)
            s += char('a' + f % 8), s += 'x';
        s += square(t);
        if (promotion(m))
            s += '=', s += "PNBRQK"[promotion(m)];
    } else {
        s += "PNBRQK"[p];
        MoveList legal;
        board.legal_moves(legal);
        bool same_file = false, same_rank = false, other = false;
        for (auto o : legal)
            if (o != m && to(o) == t && type(board.squares[from(o)]) == p) {
                other = true;
                same_file |= from(o) % 8 == f % 8;
                same_rank |= from(o) / 8 == f / 8;
            }
        if (other) {
            if (!same_file)
                s += char('a' + f % 8);
            else if (!same_rank)
                s += char('1' + f / 8);
            else
                s += square(f);
        }
        if (m & CAPTURE)
            s += 'x';
        s += square(t);
    }
    Undo u;
    if (board.make(m, u)) {
        if (board.in_check())
            s += board.has_legal_move() ? '+' : '#';
        board.unmake(m, u);
    }
    return s;
}
} // namespace dis
