#!/bin/sh

FILE=$1

echo "Note: available models are: clevr"
echo "Specified [$FILE]"

if [ "$FILE" != 'clevr' ]; then
	echo "Only clevr model is available"
	exit 1
fi

mkdir -p ./checkpoints/${FILE}_monet_pretrained

ATTN_MODEL_FILE=./checkpoints/${FILE}_monet_pretrained/latest_net_Attn.pth
CVAE_MODEL_FILE=./checkpoints/${FILE}_monet_pretrained/latest_net_CVAE.pth

g_download() {
	gURL="$1"
	dst="$2"

	# match more than 26 word characters
	ggID=$(echo "$gURL" | egrep -o '(\w|-){26,}')

	ggURL='https://drive.google.com/uc?export=download'

	curl -sc /tmp/gcokie "${ggURL}&id=${ggID}" >/dev/null
	getcode="$(awk '/_warning_/ {print $NF}' /tmp/gcokie)"

	echo -e "Downloading from "$gURL"...\n"
	(cd $(dirname "$dst") && curl --insecure -C - -LOJb /tmp/gcokie "${ggURL}&confirm=${getcode}&id=${ggID}" -o "$dst")
}

g_download "https://drive.google.com/open?id=1S4p3WL7QB67C9h91--B1G4XNIp2VrauE" "$ATTN_MODEL_FILE"
g_download "https://drive.google.com/open?id=1fA8ODaXhQE1rySH_L8PRVeMVjFG0pbQk" "$CVAE_MODEL_FILE"

shasums="$(mktemp)"
cat > "$shasums" <<EOF
0d2aeaac7dcc19181aeb84555b26ce51fe8aac2d  ./checkpoints/${FILE}_monet_pretrained/latest_net_Attn.pth
3a4f2cc31147f4a12b7b52623574e9a1ac8ed056  ./checkpoints/${FILE}_monet_pretrained/latest_net_CVAE.pth
EOF
sha1sum -c "$shasums"
rm "$shasums"
